# /lightning_module.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from nnbattle.agents.alphazero.network import Connect4Net  # Ensure correct import if needed
import logging
import matplotlib.pyplot as plt
from nnbattle.utils.tensor_utils import TensorManager

logger = logging.getLogger(__name__)

class ConnectFourLightningModule(pl.LightningModule):
    def __init__(self, agent):
        super().__init__()
        self.model = agent.model
        self.save_hyperparameters(ignore=['model'])  # Ignore model to prevent serialization issues
        self.automatic_optimization = False

        # Register input/output dimensions
        self.state_dim = 3
        self.action_dim = 7
        self.value_dim = 1

        # Initialize metrics
        self.train_metrics = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }
        
        self.loss_fn = self.loss_function
        self._train_dataloader = None
        self._val_dataloader = None
        
        # Disable validation by default for speed
        self.should_validate = False
        
        # Add performance configurations
        torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set larger chunk sizes for better GPU utilization
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of GPU memory
        
        # Enable larger tensor operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        
        # Pre-allocate memory for better performance
        torch.cuda.empty_cache()
        torch.cuda.memory.empty_cache()
        
        # Reserve a pool of memory
        self._reserve_memory = torch.empty(int(2e9), dtype=torch.float32, device='cuda')  # Reserve ~8GB
        del self._reserve_memory  # Release but keep allocated

        # Remove CPU thread limitation
        # torch.set_num_threads(1)

        self.epoch_losses = []
        self.last_gpu_log = 0

    def on_fit_start(self):
        """Called when fit begins."""
        if self.trainer.datamodule is not None:
            self._train_dataloader = self.trainer.datamodule.train_dataloader()
            self._val_dataloader = self.trainer.datamodule.val_dataloader()

    def forward(self, x):
        # Ensure x has shape [batch_size, 3, 6, 7]
        assert x.shape[1:] == (3, 6, 7), f"Input tensor has incorrect shape: {x.shape}"
        # Use self.model instead of self.agent.model
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Handle training step with proper tuple unpacking."""
        if not isinstance(batch, (list, tuple)) or len(batch) != 3:
            raise ValueError(f"Expected 3-tuple batch, got {type(batch)} with len {len(batch) if hasattr(batch, '__len__') else 'N/A'}")
            
        opt = self.optimizers()
        
        # Enable automatic mixed precision for better memory usage
        with torch.cuda.amp.autocast():
            # Correctly unpack the three values from batch
            states, mcts_probs, rewards = TensorManager.prepare_batch(*batch)
            
            # Ensure all tensors are on the correct device
            states = states.to(self.device)
            mcts_probs = mcts_probs.to(self.device)
            rewards = rewards.to(self.device)

            # Forward pass
            logits, values = self(states)
            
            # Calculate losses with adjusted weights
            value_loss = F.mse_loss(values.squeeze(-1), rewards)
            policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits + 1e-8, dim=1), dim=1))
            loss = 0.5 * value_loss + policy_loss
            
            # Manual backward and optimization
            self.manual_backward(loss)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            opt.step()
            
            # Log losses
            self.log('value_loss', value_loss, on_step=True, prog_bar=True)
            self.log('policy_loss', policy_loss, on_step=True, prog_bar=True)
            self.log('total_loss', loss, on_step=True, prog_bar=True)
            
            return {'loss': loss.item()}

    def on_train_epoch_end(self):
        """Log epoch-level metrics and GPU stats."""
        epoch = self.current_epoch
        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0
        
        logger.info(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
        self.log_gpu_stats()
        
        # Reset epoch losses
        self.epoch_losses = []
        
        # Clear GPU cache periodically
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get metrics that were logged during training steps
        metrics = self.trainer.callback_metrics
        
        # Log epoch-level metrics if they exist
        if 'loss' in metrics:
            self.log('train_loss_epoch', metrics['loss'], on_epoch=True, prog_bar=True)
        if 'train_loss' in metrics:
            self.log('train_loss_epoch', metrics['train_loss'], on_epoch=True, prog_bar=True)

        # Log epoch losses
        self.log('train_loss_epoch', avg_loss, on_epoch=True, prog_bar=True)

        # Monitor training and validation loss
        self.monitor_training()

    def monitor_training(self):
        """Monitor and visualize training and validation loss."""
        # Assuming self.trainer.callback_metrics contains validation loss
        val_loss = self.trainer.callback_metrics.get('val_loss', None)
        if val_loss is not None:
            logger.info(f"Validation Loss: {val_loss:.4f}")

        # Plot losses if needed (optional)
        # plt.plot(self.trainer.logged_metrics['train_loss_epoch'], label='Train Loss')
        # if val_loss is not None:
        #     plt.plot(val_loss, label='Validation Loss')
        # plt.legend()
        # plt.show()

    def log_gpu_stats(self):
        """Log GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"GPU Memory [GB] - Allocated: {allocated:.2f}, Reserved: {reserved:.2f}, Peak: {max_memory:.2f}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
            amsgrad=True,
            eps=1e-7,  # Smaller epsilon for better numerical stability
        )
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def loss_function(self, outputs, targets_policy, targets_value):
        logits, values = outputs
        value_loss = F.mse_loss(values.squeeze(), targets_value)
        policy_loss = -torch.mean(torch.sum(targets_policy * F.log_softmax(logits, dim=1), dim=1))
        return value_loss + policy_loss

    def validation_step(self, batch, batch_idx):
        """Handle validation step with proper unpacking."""
        # Store original mode
        was_training = self.model.training
        self.model.eval()
        
        try:
            # Correctly unpack three values
            states, mcts_probs, rewards = TensorManager.prepare_batch(*batch)
            
            # Forward pass with gradients disabled
            with torch.no_grad():
                logits, values = self(states)
                values = values.squeeze(-1)
                
                # Calculate validation metrics
                value_loss = F.mse_loss(values, rewards)
                policy_loss = -torch.mean(torch.sum(mcts_probs * F.log_softmax(logits + 1e-8, dim=1), dim=1))
                total_loss = value_loss + policy_loss
                
                self.log('val_value_loss', value_loss, on_step=False, on_epoch=True)
                self.log('val_policy_loss', policy_loss, on_step=False, on_epoch=True)
                self.log('val_loss', total_loss, on_step=False, on_epoch=True)
                
                return {'val_loss': total_loss}
        finally:
            # Restore original mode
            self.model.train(was_training)

    # Remove unnecessary hooks and callbacks
    def on_train_start(self): pass
    def on_train_end(self): pass
    def on_validation_start(self): pass
    def on_validation_end(self): pass

# Ensure __all__ is defined for easier imports
__all__ = ['ConnectFourLightningModule']