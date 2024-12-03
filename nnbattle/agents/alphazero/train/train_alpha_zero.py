from nnbattle.game.connect_four_game import ConnectFourGame
from nnbattle.constants import RED_TEAM, YEL_TEAM
from ....utils.logger_config import logger  # Add this import
import numpy as np  # Add this import
import logging
import sys

# Set up logging to output to console at INFO level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def evaluate_agent(agent, num_games=5, temperature=0.1):
    """Evaluate agent performance against random opponent."""
    wins = 0
    draws = 0
    losses = 0
    
    for game_num in range(num_games):
        game = ConnectFourGame()
        current_team = RED_TEAM  # Always start with RED_TEAM
        
        while game.get_game_state() == "ONGOING":
            try:
                # Get valid moves
                valid_moves = game.get_valid_moves()
                if not valid_moves:
                    break
                    
                # Decide which player moves
                if current_team == agent.team:
                    # Agent's turn
                    action, _ = agent.select_move(game, current_team, temperature=temperature)
                else:
                    # Random opponent's turn
                    action = np.random.choice(valid_moves)
                
                # Make move and switch teams
                game.make_move(action, current_team)
                current_team = YEL_TEAM if current_team == RED_TEAM else RED_TEAM
                
            except Exception as e:
                logger.error(f"Error during evaluation game {game_num}: {e}")
                break
        
        # Process game result
        result = game.get_game_state()
        if result == agent.team:
            wins += 1
        elif result == "Draw":
            draws += 1
        else:
            losses += 1
            
    win_rate = wins / num_games if num_games > 0 else 0.0
    logger.info(f"Evaluation results: Wins: {wins}, Draws: {draws}, Losses: {losses}")
    return win_rate

def train_alphazero(
    agent,
    max_iterations,
    num_self_play_games,
    initial_mcts_simulations_per_move,  # Renamed parameter
    max_mcts_simulations_per_move,      # Renamed parameter
    simulation_increase_interval,       # Parameter for increasing simulations
    num_evaluation_games,
    evaluation_frequency,
    use_gpu,
    save_checkpoint,
    checkpoint_frequency,
    data_module_params
):
    """Trains the AlphaZero agent using self-play and reinforcement learning."""
    from ....utils.logger_config import logger, set_log_level
    import logging
    import torch

    # Set global log level at the start of your program
    set_log_level(logging.INFO)

    # Enable tensor cores for better performance on CUDA devices
    if use_gpu and torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        logger.info("Enabled medium-precision tensor cores for CUDA device")

    import os
    import time
    from datetime import timedelta
    import numpy as np
    import pytorch_lightning as pl
    from nnbattle.game.connect_four_game import ConnectFourGame, InvalidMoveError, InvalidTurnError
    from ..agent_code import initialize_agent  # Moved import here
    from nnbattle.constants import RED_TEAM, YEL_TEAM
    from ..data_module import ConnectFourDataModule, ConnectFourDataset  # Moved import here
    from ..lightning_module import ConnectFourLightningModule
    from ..utils.model_utils import (
        load_agent_model,
        save_agent_model
    )
    from nnbattle.agents.alphazero.self_play import SelfPlay  # Import from new location if needed

    # Remove any existing logging configuration
    import signal
    def signal_handler(signum, frame):
        logger.info("\nReceived shutdown signal. Saving current state...")
        if hasattr(self_play, '_interrupt_requested'):
            self_play._interrupt_requested = True
        
        # Save current model state even if only partially trained
        interrupted_path = "mnt/ramdisk/interrupted_model.pth"
        torch.save(agent.model.state_dict(), interrupted_path)
        logger.info(f"Saved interrupted model to {interrupted_path}")
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    def log_gpu_info(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9

            logger.info(f"GPU Memory - Allocated: {allocated:.2f} GB")
            logger.info(f"GPU Memory - Reserved: {reserved:.2f} GB")
            logger.info(f"GPU Memory - Peak: {max_memory:.2f} GB")

            torch.cuda.reset_peak_memory_stats()

    # Use provided data_module_params or default values
    if data_module_params is None:
        data_module_params = {
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True
        }

    logger.info("Initializing data module...")
    # Modify DataLoader settings for better training
    data_module = ConnectFourDataModule(
        agent, 
        num_games=num_self_play_games,
        batch_size=data_module_params.get('batch_size', 1024),
        num_workers=data_module_params.get('num_workers', os.cpu_count()),
        persistent_workers=True
    )
    logger.info("Data module initialized.")

    logger.info("Initializing lightning module...")
    lightning_module = ConnectFourLightningModule(agent)  # Create the lightning module instance

    logger.info("Creating self-play instance...")
    game = ConnectFourGame()
    self_play = SelfPlay(
        game=game,
        model=agent.model,
        num_simulations=10,  # Reduced for testing
        agent=agent
    )
    
    logger.info("Starting training loop...")
    try:
        training_data = self_play.generate_training_data(num_self_play_games)
        logger.info(f"Generated {len(training_data)} training examples")
        
        if not training_data:
            logger.error("No training data generated")
            return
            
        data_module.dataset = ConnectFourDataset(training_data)
        logger.info(f"Dataset created with {len(data_module.dataset)} examples")
        
        data_module.setup('fit')
        logger.info("Data module setup completed")
        
    except Exception as e:
        logger.error(f"Error during training initialization: {e}")
        raise

    # Check if we got valid training data
    if not training_data:
        logger.error("No valid training data generated")
        raise ValueError("No valid training data generated")

    data_module.dataset = ConnectFourDataset(training_data)

    logger.info(f"Generated {len(data_module.dataset)} training examples.")

    data_module.setup('fit')

    # Create trainer with fixed configuration
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator='gpu' if use_gpu and torch.cuda.is_available() else 'cpu',
        devices=torch.cuda.device_count() if use_gpu and torch.cuda.is_available() else os.cpu_count(),
        strategy='ddp' if torch.cuda.device_count() > 1 else 'auto',
        precision=16 if use_gpu and torch.cuda.is_available() else 32,
        benchmark=True,
        deterministic=False,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        sync_batchnorm=True if torch.cuda.device_count() > 1 else False,
        profiler=None,
        enable_checkpointing=save_checkpoint,
        callbacks=[],
    )

    # Add manual early stopping logic
    best_loss = float('inf')
    patience_counter = 0

    # Add manual progress reporting
    def log_progress(iteration, total_iterations, elapsed_time):
        if iteration % 1 == 0:  # Log every iteration
            logger.info(f"Progress: {iteration}/{total_iterations} iterations "
                       f"({iteration/total_iterations*100:.1f}%) "
                       f"[{elapsed_time:.1f}s elapsed]")

    # Create a separate checkpoint saver that doesn't interfere with barebones mode
    def save_checkpoint_manually(agent, iteration, performance):
        if performance > best_performance:
            save_agent_model(agent)
            logger.info(f"Manually saved checkpoint at iteration {iteration}")

    # Disable model loading in agent's select_move during training
    agent.load_model_flag = False

    best_performance = 0.0
    no_improvement_count = 0
    best_win_rate = 0.0

    try:
        # Set proper CUDA tensor sharing strategy
        if use_gpu and torch.cuda.is_available():
            torch.multiprocessing.set_sharing_strategy('file_system')
        
        for iteration in range(1, max_iterations + 1):
            performance = None  # Initialize performance at start of each iteration
            # Adjust temperature parameter
            if iteration < max_iterations * 0.5:
                temperature = 1.0  # High temperature for more exploration
            else:
                temperature = 0.1  # Low temperature for more exploitation

            # Adjust c_puct parameter
            agent.c_puct = 1.4 if iteration < max_iterations * 0.5 else 1.0

            # Alternate starting team each iteration
            starting_team = RED_TEAM if iteration % 2 == 1 else YEL_TEAM

            # Update number of MCTS simulations progressively
            current_mcts_simulations_per_move = min(
                initial_mcts_simulations_per_move + (iteration // simulation_increase_interval),
                max_mcts_simulations_per_move
            )
            agent.mcts_simulations_per_move = current_mcts_simulations_per_move
            logger.info(f"Using {current_mcts_simulations_per_move} MCTS simulations per move for iteration {iteration}")
            
            # Log training parameters
            logger.info(f"=== Starting Training Iteration {iteration}/{max_iterations} ===")
            logger.info(f"MCTS Simulations: {current_mcts_simulations_per_move}")
            logger.info(f"Temperature: {temperature}, c_puct: {agent.c_puct}")
            
            logger.info(f"=== Starting Training Iteration {iteration}/{max_iterations} ===")
            logger.info(f"Using temperature: {temperature}, c_puct: {agent.c_puct}")
            logger.info(f"Starting team: {starting_team}")
            iteration_start_time = time.time()
            try:
                logger.info(f"Generating {num_self_play_games} self-play games...")
                log_gpu_info(agent)  # Before generating self-play games
                data_module.generate_self_play_games(temperature=temperature)
                logger.info("Self-play games generated.")

                if len(data_module.dataset) == 0:
                    # Handle the case where no data was generated
                    logger.error("No data generated during self-play. Skipping this iteration.")
                    continue

                logger.info("Starting training...")
                trainer.fit(lightning_module, datamodule=data_module)
                logger.info("Training completed.")

                # Inside the training loop, after trainer.fit:
                train_results = trainer.fit(lightning_module, datamodule=data_module)
                current_loss = lightning_module.last_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    save_agent_model(agent)
                    logger.info(f"New best loss: {best_loss}. Model saved.")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("Manual early stopping triggered.")
                        break

                # Optionally evaluate the model
                if iteration % 1 == 0:  # Evaluate every iteration
                    # Ensure CUDA tensors are properly handled
                    with torch.cuda.device(agent.device):
                        performance = evaluate_agent(agent, num_games=5)
                        torch.cuda.empty_cache()  # Clear cache after evaluation
                    
                    logger.info(f"Iteration {iteration}: Evaluation Performance: {performance}")
                    if performance > best_performance:
                        best_performance = performance
                        no_improvement_count = 0
                        if save_checkpoint:
                            # Save both checkpoint and best model
                            checkpoint_path = f"mnt/ramdisk/alphazero_model_checkpoint_{iteration}.pth"
                            best_model_path = "mnt/ramdisk/alphazero_model_best.pth"
                            torch.save(agent.model.state_dict(), checkpoint_path)
                            torch.save(agent.model.state_dict(), best_model_path)
                            logger.info(f"New best performance: {best_performance}. Models saved.")
                    else:
                        no_improvement_count += 1
                        logger.info(f"No improvement in performance. ({no_improvement_count}/{patience})")
                        if no_improvement_count >= patience:
                            logger.info("Early stopping triggered due to no improvement.")
                            break

                # Evaluate and save more frequently
                if iteration % evaluation_frequency == 0:
                    win_rate = evaluate_agent(agent, num_evaluation_games)
                    logger.info(f"Iteration {iteration}: Win Rate = {win_rate:.2f}")
                    
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_model_path = f"mnt/ramdisk/best_model_{win_rate:.2f}.pth"
                        torch.save(agent.model.state_dict(), best_model_path)
                        logger.info(f"New best model saved with win rate {win_rate:.2f}")

                # Save checkpoint regularly
                if save_checkpoint and iteration % checkpoint_frequency == 0:
                    checkpoint_path = f"mnt/ramdisk/checkpoint_{iteration}.pth"
                    torch.save(agent.model.state_dict(), checkpoint_path)
                    logger.info(f"Checkpoint saved at iteration {iteration}")

            except (InvalidMoveError, InvalidTurnError) as e:
                logger.error(f"Game error during iteration {iteration}: {e}")
                continue
            except Exception as e:
                logger.error(f"An unexpected error occurred during iteration {iteration}: {e}")
                # Clear CUDA cache on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            finally:
                logger.info(f"Iteration {iteration} completed in {time.time() - iteration_start_time:.2f} seconds")
                logger.info(f"=== Completed Iteration {iteration} ===")
                if use_gpu and torch.cuda.is_available():
                    lightning_module.log_gpu_stats()

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise
    finally:
        # Ensure proper cleanup of CUDA resources
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save the trained model
        save_agent_model(agent)

        # Copy the ramdisk contents back to the hard drive
        import shutil
        ramdisk_path = '/mnt/ramdisk/'
        hard_drive_path = 'nnbattle/agents/alphazero/model'
        shutil.copytree(ramdisk_path, hard_drive_path, dirs_exist_ok=True)
        logger.info(f"Copied ramdisk contents from {ramdisk_path} to {hard_drive_path}")

        logger.info("=== Training Completed Successfully ===")
        logger.info("Training completed. Final model saved.")

    logger.info("=== Training Completed Successfully ===")

# Ensure __all__ is defined
__all__ = ['train_alphazero']
