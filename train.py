import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import os
from nnbattle.agents.alphazero.agent_code import AlphaZeroAgent
from nnbattle.agents.alphazero.train.train_alpha_zero import train_alphazero
from nnbattle.constants import RED_TEAM
from nnbattle.utils.logger_config import logger  # Corrected import
from nnbattle.agents.alphazero.train import evaluate_agent  # Added import
import torch
import torch.multiprocessing as mp
import signal
import sys
import glob  # Added import

def main():
    # Initialize the agent with the updated parameter name
    agent = AlphaZeroAgent(
        action_dim=7,
        state_dim=3,
        mcts_simulations_per_move=10,  # Renamed parameter
        c_puct=1.4,
        load_model=True,
        team=RED_TEAM
    )

    # Update training parameters with new terminology
    training_params = {
        'initial_mcts_simulations_per_move': 800,  # Renamed parameter
        'max_mcts_simulations_per_move': 2000,     # Renamed parameter
        'simulation_increase_interval': 2,
        'num_self_play_games': 10000,              # Renamed parameter
        'num_evaluation_games': 20,
        'evaluation_frequency': 1,
        'max_iterations': 200,                     # Renamed parameter
        # ... add other training parameters here ...
    }

    # Set up CPU and CUDA optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    os.environ["OMP_NUM_THREADS"] = "4"   # Allow up to 4 threads per worker
    os.environ["MKL_NUM_THREADS"] = "4"   # Allow up to 4 threads per worker
    
    # Use larger memory chunks
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'

    # Enable shared memory for faster IPC
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Create model directory and backup untrained model
    model_dir = "nnbattle/agents/alphazero/model"
    os.makedirs(model_dir, exist_ok=True)
    untrained_model_path = os.path.join(model_dir, "untrained_baseline.pth")

    # Save untrained model for comparison
    torch.save(agent.model.state_dict(), untrained_model_path)
    logger.info(f"Saved untrained model to {untrained_model_path}")

    # Test untrained model against random
    logger.info("Testing untrained model against random player...")
    initial_performance = evaluate_agent(agent, num_games=training_params['num_evaluation_games'], temperature=0.1)
    logger.info(f"Untrained model win rate: {initial_performance:.2f}")

    # Set up CUDA and multiprocessing
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    def signal_handler(signum, frame):
        logger.info("\nReceived interrupt signal. Saving current state...")
        try:
            # Save the current model state
            interrupted_path = "mnt/ramdisk/interrupted_model.pth"
            torch.save(agent.model.state_dict(), interrupted_path)
            logger.info(f"Saved interrupted model to {interrupted_path}")

            # Clean up resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Cleanup complete")
        finally:
            sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Add checkpoint directory
    checkpoint_dir = "mnt/ramdisk/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set data module parameters
    data_module_params = {
        'batch_size': 1024,
        'num_workers': os.cpu_count(),
        'persistent_workers': True,
    }

    try:
        # Run training with GPU optimized parameters
        train_alphazero(
            agent=agent,
            max_iterations=training_params['max_iterations'],
            num_self_play_games=training_params['num_self_play_games'],
            initial_mcts_simulations_per_move=training_params['initial_mcts_simulations_per_move'],
            max_mcts_simulations_per_move=training_params['max_mcts_simulations_per_move'],
            simulation_increase_interval=training_params['simulation_increase_interval'],
            num_evaluation_games=training_params['num_evaluation_games'],
            evaluation_frequency=training_params['evaluation_frequency'],
            use_gpu=True,
            save_checkpoint=True,
            checkpoint_frequency=1,
            data_module_params=data_module_params
        )
        
        # Test trained model
        logger.info("Testing trained model against random player...")
        final_performance = evaluate_agent(agent, num_games=training_params['num_evaluation_games'])
        logger.info(f"Training Results:")
        logger.info(f"Initial win rate: {initial_performance:.2f}")
        logger.info(f"Final win rate: {final_performance:.2f}")
        logger.info(f"Improvement: {final_performance - initial_performance:.2f}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        signal_handler(signal.SIGTERM, None)  # Clean up on error

if __name__ == "__main__":
    main()