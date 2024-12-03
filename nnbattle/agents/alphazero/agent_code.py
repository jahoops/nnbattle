# /agent_code.py

import logging

# Configure logging at the very start
logging.basicConfig(
    level=logging.DEBUG,  # Or DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Proceed with your imports
import os
import time
from datetime import timedelta
import torch
import pytorch_lightning as pl
import numpy as np
import copy
from nnbattle.game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from nnbattle.constants import RED_TEAM, YEL_TEAM, EMPTY  # Add EMPTY to imports
from .network import Connect4Net
from .utils.model_utils import load_agent_model, save_agent_model
from .mcts import MCTSNode, mcts_simulate
from ..base_agent import BaseAgent  # Adjusted import
from nnbattle.utils.logger_config import logger
from nnbattle.utils.tensor_utils import TensorManager

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

from contextlib import contextmanager

@contextmanager
def model_mode(model, training):
    original_mode = model.training
    model.train(training)
    try:
        yield
    finally:
        model.train(original_mode)

def check_model_initialization(model):
    """Check model initialization with more appropriate thresholds."""
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        mean_abs = param.data.abs().mean().item()
        
        if 'bias' in name:
            # Biases should be close to zero
            if mean_abs > 0.1:  # Relaxed threshold for biases
                logger.warning(f"Warning: Bias initialization issue in layer {name} (mean abs: {mean_abs:.4f})")
        elif 'weight' in name:
            if 'bn' in name:  # BatchNorm weights
                if abs(mean_abs - 1.0) > 0.1:  # Should be close to 1
                    logger.warning(f"Warning: BatchNorm weight initialization issue in layer {name} (mean abs: {mean_abs:.4f})")
            elif 'fc' in name or 'conv' in name:  # FC and Conv layers
                if mean_abs > 2.0 or mean_abs < 0.01:  # Reasonable range for Kaiming init
                    logger.warning(f"Warning: Weight initialization issue in layer {name} (mean abs: {mean_abs:.4f})")

class AlphaZeroAgent(BaseAgent):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        action_dim,
        state_dim,
        mcts_simulations_per_move,
        c_puct,
        load_model,
        team,
        model_path
    ):
        super().__init__(team)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda")  # Always CUDA
        self.model_loaded = False  # Initialize as False
        self.load_model_flag = load_model
        self.mcts_simulations_per_move = mcts_simulations_per_move
        self.c_puct = c_puct
        self.team = team
        self.memory = []
        self.model_path = model_path if model_path else "mnt/ramdisk/alphazero_model_final.pth"  # Use provided path or default

        # Initialize the model directly on CUDA with verification
        self.model = Connect4Net(state_dim, action_dim).cuda()
        
        # Verify CUDA placement
        for param in self.model.parameters():
            if not param.is_cuda:
                logger.error(f"Parameter {param.shape} not on CUDA!")
                param.data = param.data.cuda()

        # Check model initialization
        check_model_initialization(self.model)

        if load_model:
            try:
                # Use weights_only=True for safety and to prevent pickle warnings
                state_dict = torch.load(
                    self.model_path, 
                    map_location='cuda',
                    weights_only=True  # Add this parameter
                )
                # Force state dict to CUDA
                state_dict = {k: v.cuda() for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                
                # Verify again after loading
                for param in self.model.parameters():
                    if not param.is_cuda:
                        logger.error(f"Parameter moved to CPU after loading!")
                        param.data = param.data.cuda()
                
                self.model_loaded = True
                logger.info("Model loaded and verified on CUDA")
            except FileNotFoundError:
                logger.warning("No model file found, starting with fresh model on CUDA")
                self.model_loaded = False
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model_loaded = False

        # Log initialization after attempting to load the model
        logger.info(f"Initialized AlphaZeroAgent on device: {self.device}")
        logger.info("AlphaZeroAgent setup complete.")

        # After model initialization and model loading
        logger.debug(f"After initialization, self.model is {self.model}")
        logger.debug(f"Model type: {type(self.model)}")

        # Check if self.model is None
        if self.model is None:
            logger.error("Agent model is None after initialization.")
        else:
            logger.info("Agent model is properly initialized.")

        # Add periodic CUDA verification
        self.verify_cuda_state()

    def __getstate__(self):
        """Customize pickling behavior."""
        state = self.__dict__.copy()
        # Remove non-picklable entries if any
        if 'logger' in state:
            del state['logger']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-initialize logger or other non-picklable attributes if necessary

    def verify_cuda_state(self):
        """Verify all model components are on CUDA."""
        if not next(self.model.parameters()).is_cuda:
            logger.error("Model not on CUDA! Forcing CUDA placement.")
            self.model.cuda()
        
        for name, param in self.model.named_parameters():
            if not param.is_cuda:
                logger.error(f"Parameter {name} not on CUDA! Forcing CUDA placement.")
                param.data = param.data.cuda()

    def save_model(self):
        """Save the agent's model to the specified model path."""
        torch.save(
            self.model.state_dict(), 
            self.model_path,
            _use_new_zipfile_serialization=True  # Use new format
        )
        logger.info(f"Model saved to {self.model_path}")

    def preprocess(self, board, team):
        board = board.copy()
        
        # Create a 3-channel state representation with proper normalization
        current_board = (board == team).astype(np.float32)
        opponent_board = (board == (3 - team)).astype(np.float32)
        valid_moves = np.zeros_like(current_board)
        
        for col in range(board.shape[1]):
            for row in range(board.shape[0] - 1, -1, -1):
                if board[row][col] == EMPTY:
                    valid_moves[row][col] = 1.0  # Ensure float value
                    break
        
        # Stack and normalize channels
        state = np.stack([
            current_board,  # Already binary [0,1]
            opponent_board,  # Already binary [0,1]
            valid_moves     # Already binary [0,1]
        ])
        
        # Double-check normalization
        assert np.all((state >= 0) & (state <= 1)), "State values outside [0,1] range"
        
        # Convert to tensor
        return TensorManager.numpy_to_tensor(state)

    def select_move(self, game: ConnectFourGame, team: int, temperature=1.0):
        """Select a valid move for the current game state."""
        if game.get_game_state() != "ONGOING":
            raise InvalidMoveError("Game is not ongoing.")
            
        # Get valid moves first
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise InvalidMoveError("No valid moves available")
            
        try:
            # Remove unnecessary turn check
            # if game.last_team == team:
            #     raise InvalidTurnError(f"Invalid turn: team {team} cannot move after itself")
                
            selected_action, action_probs = self.act(game, team, temperature=temperature)
            
            # Ensure selected action is valid
            if (selected_action not in valid_moves):
                logger.warning(f"Selected invalid move {selected_action}, falling back to random valid move")
                selected_action = np.random.choice(valid_moves)
                
            # Test move validity before returning
            game_copy = deepcopy_env(game)
            game_copy.make_move(selected_action, team)
            
            self.memory.append((self.preprocess(game.get_board(), team), action_probs, 0.0))
            return selected_action, action_probs
            
        except (InvalidMoveError, InvalidTurnError) as e:
            logger.error(f"Error selecting move: {e}")
            if valid_moves:  # If we have valid moves, make a random one as fallback
                return np.random.choice(valid_moves), torch.zeros(self.action_dim)
            raise

    def evaluate_model(self):
        """Evaluate the model on a set of validation games to monitor learning progress."""
        logger.info("Starting model evaluation.")
        # Implement evaluation logic, e.g., play against a random agent
        wins = 0
        draws = 0
        losses = 0
        num_evaluations = 20
        for _ in range(num_evaluations):
            game = ConnectFourGame()
            while game.get_game_state() == "ONGOING":
                if game.last_team == self.team:
                    action, _ = self.select_move(game, temperature=0)
                else:
                    valid_moves = game.get_valid_moves()
                    action = np.random.choice(valid_moves)
                game.make_move(action, game.last_team if game.last_team else RED_TEAM)
            result = game.get_game_state()
            if result == self.team:
                wins += 1
            elif result == "Draw":
                draws += 1
            else:
                losses += 1
        logger.info(f"Evaluation Results over {num_evaluations} games: Wins={wins}, Draws={draws}, Losses={losses}")

    def act(self, game: ConnectFourGame, team: int, temperature=1.0, **kwargs):
        """Use MCTS to select moves in both training and tournament play."""
        # Ensure temperature is not zero
        temperature = max(temperature, 1e-8)  # Add small epsilon to prevent zero
        logger.debug(f"Acting with temperature {temperature}")
        
        # Verify CUDA state before acting
        self.verify_cuda_state()
        
        try:
            with model_mode(self.model, False):  # Ensure model is in eval mode
                selected_action, action_probs = mcts_simulate(
                    self, 
                    game, 
                    team, 
                    temperature=temperature
                )
                return selected_action, action_probs
        except Exception as e:
            logger.error(f"Error in MCTS simulation: {e}")
            valid_moves = game.get_valid_moves()
            if valid_moves:
                return np.random.choice(valid_moves), torch.zeros(self.action_dim)
            raise

def initialize_agent(
    action_dim,
    state_dim,
    mcts_simulations_per_move,
    c_puct,
    load_model,
    team,
    model_path
) -> AlphaZeroAgent:
    return AlphaZeroAgent(
        action_dim=action_dim,
        state_dim=state_dim,
        mcts_simulations_per_move=mcts_simulations_per_move,
        c_puct=c_puct,
        load_model=load_model,
        team=team,
        model_path=model_path
    )

__all__ = ['AlphaZeroAgent', 'initialize_agent']