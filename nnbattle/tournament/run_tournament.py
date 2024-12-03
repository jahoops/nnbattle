# FILE: tournament/run_tournament.py

import json
import os
import logging
from typing import List

from nnbattle.agents.alphazero import AlphaZeroAgent
from nnbattle.agents.minimax.agent_code import MinimaxAgent  # Ensure correct import path
from nnbattle.game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
from nnbattle.constants import RED_TEAM, YEL_TEAM
from nnbattle.agents.base_agent import BaseAgent

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths to the models
FINAL_MODEL_PATH = 'mnt/ramdisk/alphazero_model_final.pth'      # Replace with actual path
BASELINE_MODEL_PATH = 'mnt/ramdisk/alphazero_model_baseline.pth'  # Replace with actual path

def run_tournament(agents: List[BaseAgent], num_games=10):
    # Format agent names with team strings instead of numbers
    results = {f"{agent.__class__.__name__} ({'RED_TEAM' if agent.team == RED_TEAM else 'YEL_TEAM'})": 0 for agent in agents}
    results['draws'] = 0
    game = ConnectFourGame()

    for i in range(num_games):
        game.reset()
        current_team = RED_TEAM if i % 2 == 0 else YEL_TEAM  # Alternate starting team
        logger.info(f"Starting Game {i+1}: Team {current_team} starts")

        while game.get_game_state() == "ONGOING":
            agent = next((a for a in agents if a.team == current_team), None)
            if agent is None:
                logger.error(f"No agent found for team {current_team}")
                break
                
            try:
                # Handle different agent types
                if isinstance(agent, AlphaZeroAgent):
                    move_result = agent.select_move(game, agent.team)
                else:
                    move_result = agent.select_move(game)
                    
                selected_action = move_result[0] if isinstance(move_result, tuple) else move_result
                logger.debug(f"Agent {agent.__class__.__name__} ({agent.team}) selects column {selected_action}")
                
                move_successful = game.make_move(selected_action, agent.team)
                if move_successful:
                    logger.info(f"Team {agent.team} placed piece in column {selected_action}")
                    logger.info(f"Board state:\n{game.board_to_string()}")
                else:
                    logger.error(f"Move unsuccessful for team {agent.team} in column {selected_action}")
                    break
            except (InvalidMoveError, InvalidTurnError) as e:
                logger.error(f"Invalid move by {agent.__class__.__name__}: {e}")
                break

            current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM

        # Log game result
        result = game.get_game_state()
        logger.info(f"Game {i+1} ended with result: {result}")
        logger.info(f"Final board:\n{game.board_to_string()}")
        
        if result in [RED_TEAM, YEL_TEAM]:
            winner = next((a for a in agents if a.team == result), None)
            if winner:
                team_str = 'RED_TEAM' if result == RED_TEAM else 'YEL_TEAM'
                results[f"{winner.__class__.__name__} ({team_str})"] += 1
        else:
            results['draws'] += 1

    # Save results
    results_dir = os.path.join('tournament', 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'tournament_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("Tournament completed. Results saved to tournament/results/tournament_results.json")

if __name__ == "__main__":
    # Initialize Baseline Agent (e.g., YEL_TEAM)
    agent_baseline = MinimaxAgent(depth=2, team=YEL_TEAM)
    """agent_baseline = AlphaZeroAgent(
        action_dim=7,
        state_dim=3,
        use_gpu=True,
        num_simulations=0,
        c_puct=1.4,
        load_model=True,
        model_path=BASELINE_MODEL_PATH,  # Load baseline model
        team=YEL_TEAM  # Assign YEL_TEAM
    )"""

    # Initialize Final Agent (e.g., RED_TEAM)
    agent_final = AlphaZeroAgent(
        action_dim=7,
        state_dim=3,
        num_simulations=2000,
        c_puct=1.4,
        load_model=True,
        model_path=FINAL_MODEL_PATH,  # Load final model
        team=RED_TEAM  # Assign RED_TEAM
    )

    agents = [agent_baseline, agent_final]
    run_tournament(agents, num_games=10)