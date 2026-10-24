# /mcts.py

import logging
import math
import copy
from typing import Optional, List
import signal
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F
import numpy as np
from ...game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from ...constants import RED_TEAM, YEL_TEAM
from .utils.model_utils import preprocess_board
from nnbattle.utils.tensor_utils import TensorManager

logger = logging.getLogger(__name__)

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def deepcopy_env(env):
    """Deep copy the environment, including last_team."""
    new_env = copy.deepcopy(env)
    return new_env

class MCTSNode:
    def __init__(self, parent: Optional['MCTSNode'], action: Optional[int], board: np.ndarray, player: int):
        """Initialize node with board state and player whose turn it is."""
        self.parent = parent
        self.action = action
        self.board = board.copy()
        self.player = player  # The player to move at this node
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def best_child(self, c_puct: float) -> Optional['MCTSNode']:
        best_score = -float('inf')
        best_node = None
        epsilon = 1e-8  # Small value to prevent division by zero
        for child in self.children.values():
            # This is the UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
            u = c_puct * child.prior * math.sqrt(self.visits + epsilon) / (1 + child.visits + epsilon)
            q = child.value / (1 + child.visits + epsilon)
            score = q + u  # UCB score combines exploitation (q) and exploration (u)
            if score > best_score:
                best_score = score
                best_node = child
        return best_node

    def expand(self, action_probs: torch.Tensor, legal_actions: List[int], next_player: int):
        """Expand node with valid moves and correct player turn."""
        for action in legal_actions:
            if action not in self.children:
                try:
                    next_board = self.board.copy()
                    # Simulate the action
                    next_game = ConnectFourGame()
                    next_game.board = next_board
                    next_game.make_move(action, self.player)

                    child_node = MCTSNode(
                        parent=self,
                        action=action,
                        board=next_game.get_board(),
                        player=next_player  # Set the next player
                    )
                    child_node.prior = action_probs[action]
                    self.children[action] = child_node
                except (InvalidMoveError, InvalidTurnError) as e:
                    logger.error(f"Invalid move during expansion: {e}")
                    continue

    def backpropagate(self, reward: float):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)

def mcts_simulate(agent, game: ConnectFourGame, team: int, temperature):
    """Monte Carlo Tree Search simulation with CUDA verification."""
    # Verify CUDA state at start of simulation
    agent.verify_cuda_state()
    
    valid_moves = game.get_valid_moves()
    
    # Early in training, use mostly random play for fast games
    if agent.mcts_simulations_per_move < 25:  # This is causing the fast initial games
        action = np.random.choice(valid_moves)
        policy = TensorManager.to_tensor(
            torch.zeros(agent.action_dim),
            dtype=torch.float32
        )
        policy[valid_moves] = 1.0 / len(valid_moves)
        return action, policy

    try:
        # Remove unnecessary turn check
        # if game.last_team == team:
        #     raise InvalidTurnError(f"Invalid turn: team {team} cannot move after itself")

        # Initialize the root node with the current player
        root = MCTSNode(parent=None, action=None, board=game.get_board(), player=team)
        root.visits = 1

        for sim in range(agent.mcts_simulations_per_move):
            current_node = root
            sim_game = deepcopy_env(game)

            # Selection
            while not current_node.is_leaf():
                # Select the best child
                current_node = current_node.best_child(agent.c_puct)
                if current_node is None:
                    break
                # Make the move in the simulated game
                sim_game.make_move(current_node.action, current_node.parent.player)

            # Expansion
            if sim_game.get_game_state() == "ONGOING":
                # The next player is the opposite of the current node's player
                next_player = YEL_TEAM if current_node.player == RED_TEAM else RED_TEAM

                with torch.no_grad():
                    # Use TensorManager to handle state tensor
                    state_tensor = agent.preprocess(sim_game.get_board(), next_player)
                    # Verify tensor device
                    if not state_tensor.is_cuda:
                        logger.error("State tensor not on CUDA!")
                        state_tensor = state_tensor.cuda()
                    
                    # Verify model device before forward pass
                    agent.verify_cuda_state()
                    
                    # Add batch dimension and ensure CUDA tensor
                    state_tensor = TensorManager.ensure_tensor(
                        state_tensor.unsqueeze(0)
                    )
                    action_logits, value = agent.model(state_tensor)
                    
                    # Verify output device
                    if not action_logits.is_cuda or not value.is_cuda:
                        logger.error("Model outputs not on CUDA!")
                        action_logits = action_logits.cuda()
                        value = value.cuda()
                
                # Use TensorManager for action probabilities
                action_probs = F.softmax(action_logits.squeeze(), dim=0)
                valid_moves = sim_game.get_valid_moves()
                current_node.expand(action_probs, valid_moves, next_player)

            # Simulation / Evaluation
            game_result = sim_game.get_game_state()
            if game_result == "ONGOING":
                # The value is from the perspective of the current node's player
                leaf_value = value.item()
            else:
                if game_result == current_node.player:
                    leaf_value = 1.0
                elif game_result == "Draw":
                    leaf_value = 0.0
                else:
                    leaf_value = -1.0

            # Backpropagation
            current_node.backpropagate(-leaf_value)

        # Select move based on visit counts and temperature
        valid_children = [(child.action, child.visits) for child in root.children.values()]
        if not valid_children:
            valid_moves = game.get_valid_moves()
            return np.random.choice(valid_moves), torch.zeros(agent.action_dim)

        actions, visits = zip(*valid_children)
        visits = np.array(visits, dtype=np.float32)

        # Ensure total visits are not zero to prevent division by zero
        total_visits = visits.sum()
        if total_visits == 0:
            logger.error("Total visits are zero, cannot compute probabilities")
            visits += 1e-8  # Add a small value to prevent zero division
            total_visits = visits.sum()

        # Compute probabilities safely
        probs = visits ** (1.0 / max(temperature, 1e-8))  # Prevent division by zero
        probs_sum = probs.sum()
        if probs_sum == 0 or np.isnan(probs_sum):
            logger.error("Sum of probabilities is zero or NaN, assigning uniform probabilities")
            probs = np.ones_like(probs) / len(probs)
        else:
            probs = probs / probs_sum

        # Select action based on computed probabilities
        action = np.random.choice(actions, p=probs)

        # Create final policy tensor using TensorManager
        policy = TensorManager.to_tensor(
            torch.zeros(agent.action_dim),
            dtype=torch.float32
        )
        policy[list(actions)] = TensorManager.to_tensor(
            visits / visits.sum(),
            dtype=torch.float32
        )

        return action, policy

    except Exception as e:
        logger.error(f"MCTS simulation failed: {str(e)}")
        # Log detailed device information
        logger.error(f"Model device: {next(agent.model.parameters()).device}")
        logger.error(f"Model state: {[p.is_cuda for p in agent.model.parameters()]}")
        if 'state_tensor' in locals():
            logger.error(f"Input tensor device: {state_tensor.device}")
        
        # Create fallback policy tensor using TensorManager
        policy = TensorManager.to_tensor(
            torch.zeros(agent.action_dim),
            dtype=torch.float32
        )
        
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves available")
        return np.random.choice(valid_moves), policy
