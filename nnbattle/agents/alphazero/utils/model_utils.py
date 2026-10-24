import logging
import os
import torch
import numpy as np
from typing import TYPE_CHECKING, Optional

# Use type hinting carefully to avoid circular imports
if TYPE_CHECKING:
    from ..agent_code import AlphaZeroAgent

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set appropriate logging level

def load_agent_model(agent: 'AlphaZeroAgent'):
    try:
        state_dict = torch.load(agent.model_path, map_location=agent.device)
        agent.model.load_state_dict(state_dict)
        agent.model.to(agent.device)
        agent.model_loaded = True
        agent.logger.info(f"Model loaded successfully from {agent.model_path}.")
    except FileNotFoundError:
        agent.logger.warning(f"Model file not found at {agent.model_path}.")
        agent.model_loaded = False
    except Exception as e:
        agent.logger.error(f"Error loading model from {agent.model_path}: {e}")
        agent.model_loaded = False

def save_agent_model(agent: 'AlphaZeroAgent'):
    """
    Saves the agent's model state dictionary to the specified path.

    :param agent: Instance of AlphaZeroAgent.
    :param path: Destination path for the model weights.
    """
    try:
        torch.save(agent.model.state_dict(), agent.model_path)
        logger.info(f"Model saved successfully to {agent.model_path}.")
    except Exception as e:
        logger.error(f"Failed to save the model to {agent.model_path}: {e}")

EMPTY = 0  # Define the EMPTY constant

def preprocess_board(board_state: np.ndarray, team: int) -> torch.Tensor:
    """
    Preprocesses the board state into a tensor suitable for the model.

    :param board_state: Numpy array representing the board.
    :param team: Current team's identifier.
    :return: Preprocessed tensor.
    """
    current_board = (board_state == team).astype(np.float32)
    opponent_board = (board_state == (3 - team)).astype(np.float32)
    valid_moves = np.zeros_like(current_board)
    for col in range(board_state.shape[1]):
        for row in range(board_state.shape[0]-1, -1, -1):
            if board_state[row][col] == EMPTY:
                valid_moves[row][col] = 1
                break
    state = np.stack([current_board, opponent_board, valid_moves])
    tensor = torch.from_numpy(state)
    return tensor.unsqueeze(0)  # Add batch dimension if required

__all__ = ['load_agent_model', 'save_agent_model', 'preprocess_board']