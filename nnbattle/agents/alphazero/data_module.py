import pytorch_lightning as pl
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import logging
from nnbattle.agents.alphazero.self_play import SelfPlay  # Updated import
from nnbattle.game.connect_four_game import ConnectFourGame
from nnbattle.utils.tensor_utils import TensorManager
import multiprocessing as mp

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Changed from DEBUG to INFO

# Ensure __all__ is defined
__all__ = ['ConnectFourDataset', 'ConnectFourDataModule']

class ConnectFourDataset(Dataset):
    def __init__(self, data):
        self.states = [item[0] for item in data]
        self.mcts_probs = [item[1] for item in data]
        self.rewards = [item[2] for item in data]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        # Return raw data; any tensor conversions should be done in collate_fn
        return self.states[idx], self.mcts_probs[idx], self.rewards[idx]


def check_data_integrity(data_loader):
    """Check data integrity with proper unpacking for 3-value tuples."""
    for states, mcts_probs, rewards in data_loader:
        if (states < 0).any() or (states > 1).any():
            logger.warning("Warning: State values out of expected range (0-1)")
        break  # Check the first batch for simplicity


def collate_fn(batch):
    """Custom collate function to convert batch data to tensors."""
    states, mcts_probs, rewards = zip(*batch)
    states = torch.stack([TensorManager.to_tensor(s) for s in states])
    mcts_probs = torch.stack([TensorManager.to_tensor(p) for p in mcts_probs])
    rewards = torch.tensor(rewards, dtype=torch.float32).cuda()
    return states, mcts_probs, rewards


class ConnectFourDataModule(pl.LightningDataModule):
    def __init__(self, agent, self_play_games_per_round, batch_size, num_workers, persistent_workers):
        super().__init__()
        self.agent = agent
        self.self_play_games_per_round = self_play_games_per_round
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.dataset = ConnectFourDataset([])

    def _generate_game_data(self, args):
        """Function to be executed in each process."""
        index, seed, temperature = args  # Unpack three values now
        # Set a unique seed for each process
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Create a fresh game and self-play instance
        game = ConnectFourGame()
        self_play = SelfPlay(
            game=game,
            model=self.agent.model,
            mcts_simulations_per_move=self.agent.mcts_simulations_per_move,
            agent=self.agent
        )
        print(f"Process {index}: Generating data with seed {seed}")
        # Generate training data for one game
        return self_play.generate_training_data(1, temperature=temperature, seed=seed)

    def generate_self_play_games(self, temperature=1.0):
        """Generate self-play games using multiprocessing with temperature control."""
        print(f"Generating {self.self_play_games_per_round} self-play games with temperature {temperature} and {self.agent.mcts_simulations_per_move} MCTS simulations per move.")
        try:
            num_processes = mp.cpu_count()
            print(f"Using {num_processes} processes for game simulation.")

            with mp.Pool(processes=num_processes) as pool:
                # Create a list of seeds for reproducibility
                seeds = [np.random.randint(0, 2**32 - 1) for _ in range(self.self_play_games_per_round)]
                # Include index with seed and temperature
                args = [(i, seed, temperature) for i, seed in enumerate(seeds)]
                # Start the pool of processes
                results = pool.map(self._generate_game_data, args)

            # Flatten the list of results
            training_data = [item for sublist in results for item in sublist]
            self.dataset = ConnectFourDataset(training_data)
            logger.info(f"Generated {len(self.dataset)} training examples.")
        except Exception as e:
            logger.error(f"An error occurred during self-play generation: {e}")
            raise
        self.setup('fit')

    def setup(self, stage=None):
        """Ensure data is properly split and initialized."""
        if stage == 'fit' or stage is None:
            if len(self.dataset) == 0:
                logger.error("Dataset is empty. Cannot proceed with training.")
                raise ValueError("Dataset is empty. Generate self-play games first.")
            
            # Split data into training and validation
            total = len(self.dataset)
            val_size = max(int(0.2 * total), 1)
            train_size = total - val_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
            )
            logger.info(f"Data split: {train_size} training, {val_size} validation samples")

            # Check data integrity after setting up datasets
            try:
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    num_workers=0  # Use 0 workers for debugging
                )
                check_data_integrity(train_loader)
            except Exception as e:
                logger.error(f"Data integrity check failed: {e}")
                raise

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Enable pin_memory for faster host-to-device transfer
            persistent_workers=False,  # Disable to prevent potential hanging
            prefetch_factor=2,
            collate_fn=collate_fn,  # Use custom collate function
            drop_last=True
        )

    def val_dataloader(self):
        """Create and return the validation dataloader."""
        if not hasattr(self, 'val_dataset'):
            self.setup('fit')
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers // 2,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=2,
            collate_fn=collate_fn,
            drop_last=True
        )

    def _worker_init_fn(self, worker_id: int):
        """Initialize each worker with a different seed."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)