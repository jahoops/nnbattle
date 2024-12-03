import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TensorManager:
    """Centralized tensor management with CUDA-only handling."""
    
    @staticmethod
    def to_tensor(data, dtype=torch.float32):
        """Convert input to CUDA tensor with proper type."""
        if isinstance(data, torch.Tensor):
            tensor = data.cuda()
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).cuda()
        else:
            tensor = torch.tensor(data, device='cuda')
        return tensor.to(dtype=dtype)

    @staticmethod
    def ensure_tensor(tensor, dtype=None):
        """Ensure tensor is on CUDA with correct type."""
        if not isinstance(tensor, torch.Tensor):
            return TensorManager.to_tensor(tensor, dtype or torch.float32)
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        if dtype and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor

    @staticmethod
    def numpy_to_tensor(array):
        """Convert numpy array to CUDA tensor."""
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        return torch.from_numpy(array).float().cuda()

    @staticmethod
    def prepare_batch(states, mcts_probs, rewards):
        """Prepare a batch of data for CUDA."""
        states = TensorManager.ensure_tensor(states)
        mcts_probs = TensorManager.ensure_tensor(mcts_probs)
        rewards = TensorManager.ensure_tensor(rewards, dtype=torch.float32)
        
        # Release any stale CUDA memory
        torch.cuda.empty_cache()
        
        return states, mcts_probs, rewards

    @staticmethod
    def validate_tensor(tensor, expected_shape=None, value_range=None, name="tensor"):
        """Validate tensor properties."""
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
            
        if expected_shape and tensor.shape[1:] != expected_shape:
            raise ValueError(f"{name} has wrong shape: {tensor.shape}, expected {expected_shape}")
            
        if value_range:
            min_val, max_val = value_range
            if tensor.min() < min_val or tensor.max() > max_val:
                raise ValueError(f"{name} values outside range [{min_val}, {max_val}]")

        return tensor

def collate_fn(batch):
    """Custom collate function to convert batch data to tensors."""
    states, mcts_probs, rewards = zip(*batch)
    states = torch.stack([TensorManager.to_tensor(s) for s in states])
    mcts_probs = torch.stack([TensorManager.to_tensor(p) for p in mcts_probs])
    rewards = torch.tensor(rewards, dtype=torch.float32).cuda()
    return states, mcts_probs, rewards