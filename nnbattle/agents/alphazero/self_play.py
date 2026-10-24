import numpy as np
import copy
from .mcts import mcts_simulate
from typing import List, Tuple
from ...utils.logger_config import logger
from ...game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from ...constants import RED_TEAM, YEL_TEAM

def deepcopy_env(env):
    """Deep copy the environment."""
    return copy.deepcopy(env)

class SelfPlay:
    def __init__(self, game, model, mcts_simulations_per_move, agent):
        self.game = game
        self.model = model
        self.mcts_simulations_per_move = mcts_simulations_per_move
        self.agent = agent
        # Remove or modify any attributes that cannot be pickled

    def execute_self_play_game(self, game_num: int, temperature) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Execute one self-play game with correct player turn management."""
        game = ConnectFourGame()
        states, policies, values = [], [], []
        game_history = []
        current_player = RED_TEAM  # Start with RED_TEAM

        try:
            while game.get_game_state() == "ONGOING":
                # Get valid moves
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break

                # Get move from MCTS
                action, policy = mcts_simulate(self.agent, game, current_player, temperature)

                # Make the move
                if not game.make_move(action, current_player):
                    logger.error(f"Invalid move {action} for team {current_player}")
                    break

                # Store the state and policy from the perspective of current_player
                game_history.append((
                    game.get_board().copy(),
                    policy,
                    current_player
                ))

                # Switch players
                current_player = YEL_TEAM if current_player == RED_TEAM else RED_TEAM

            # Process game result
            result = game.get_game_state()
            #logger.info(f"Game {game_num} finished with result: {result}")
            return self._process_game_history(game_history, result)

        except Exception as e:
            logger.error(f"Game {game_num} failed: {str(e)}")
            return [], [], []

    def _process_game_history(self, game_history: List[Tuple], result: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """Process game history into training data."""
        if not game_history:
            return [], [], []
            
        states, policies, values = [], [], []
        for state, policy, team in reversed(game_history):
            reward = 0.0 if result == "Draw" else (1.0 if result == team else -1.0)
            states.append(state)
            policies.append(policy)
            values.append(reward)
        
        return states, policies, values

    def generate_training_data(self, self_play_games_per_round: int, temperature=1.0, seed=0) -> List[Tuple]:
        """Generate training data through self-play with temperature control."""
        #print(f"Starting self-play data generation for {self_play_games_per_round} games (temperature={temperature})")
        training_data = []
        completed_games = 0
        
        try:
            for game in range(self_play_games_per_round):
                #print(f"Game {game + 1}/{self_play_games_per_round}")
                if hasattr(self, '_interrupt_requested'):
                    logger.info("Interrupt requested, saving current progress...")
                    break
                try:
                    states, policies, values = self.execute_self_play_game(game, temperature)
                    if states:
                        training_data.extend(zip(states, policies, values))
                        completed_games += 1
                        
                    if seed > 0:
                        print(f"Seed: {seed}, Examples: {len(training_data)}")
                    elif game % 100 == 0:
                        print(f"Progress: {game + 1}/{self_play_games_per_round} games, Examples: {len(training_data)}")
                        
                except KeyboardInterrupt:
                    logger.info("Interrupt received during game, saving progress...")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Interrupt received, saving current progress...")
        finally:
            return training_data