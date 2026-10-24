# /network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize residual block with appropriate scaling."""
        # Conv layers
        for m in [self.conv1, self.conv2]:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        # BatchNorm layers
        for m in [self.bn1, self.bn2]:
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual  # Ensure no in-place operations
        out = self.relu(out)
        return out

class Connect4Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Connect4Net, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(state_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(5)]
        )

        # Policy head
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 6 * 7, action_dim)

        # Value head
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 6 * 7, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self._initialize_weights()

        # Ensure device attribute is set
        self._device = torch.device('cuda')

        # Set appropriate default modes for different components
        self.train()  # Set default mode to train
        self.bn1.eval()  # BatchNorm layers should typically be in eval during inference
        self.policy_bn.eval()
        self.value_bn.eval()
        
        # Set dropout if used (for example)
        self.training = True  # Explicitly set training mode

        # Force CUDA at initialization
        self.cuda()
        
        # Verify CUDA state immediately after initialization
        if not next(self.parameters()).is_cuda:
            logger.error("Network not on CUDA after initialization!")
            self.cuda()  # Try forcing CUDA again
            
        # Verify each component
        for name, module in self.named_modules():
            if hasattr(module, 'weight') and not module.weight.is_cuda:
                logger.error(f"Module {name} weights not on CUDA!")
                module.cuda()

    def _initialize_weights(self):
        """Initialize network with custom scaling for different components."""
        # Conv layers initialization
        for m in [self.conv1, self.policy_conv]:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()

        # Special initialization for value conv
        n = self.value_conv.kernel_size[0] * self.value_conv.kernel_size[1] * self.value_conv.out_channels
        self.value_conv.weight.data.normal_(0, math.sqrt(1. / n))  # Smaller variance
        self.value_conv.bias.data.zero_()

        # BatchNorm layers
        for m in [self.bn1, self.policy_bn, self.value_bn]:
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        # Policy head
        n = self.policy_fc.weight.size(1)
        self.policy_fc.weight.data.normal_(0, math.sqrt(2. / n))
        self.policy_fc.bias.data.zero_()

        # Value head
        n = self.value_fc1.weight.size(1)
        self.value_fc1.weight.data.normal_(0, math.sqrt(2. / n))
        self.value_fc1.bias.data.zero_()
        
        # Final value layer - adjust initialization range to be between 0.01 and 2.0
        self.value_fc2.weight.data.normal_(0, 0.02)  # Increased from 0.01 to 0.02
        self.value_fc2.bias.data.zero_()

        # Initialize residual blocks
        for block in self.res_blocks:
            block._initialize_weights()

    def to(self, device):
        if device != 'cuda' and device != torch.device('cuda'):
            logger.warning("Attempting to move model off CUDA - forcing CUDA")
            device = torch.device('cuda')
        return super().to(device)

    def forward(self, x):
        # Verify CUDA state at start of forward pass
        if not next(self.parameters()).is_cuda:
            logger.error("Network moved to CPU before forward pass!")
            self.cuda()
        
        # Ensure batch norm layers are in correct mode during forward pass
        self.bn1.train(self.training)
        self.policy_bn.train(self.training)
        self.value_bn.train(self.training)
        
        # Ensure x is on the correct device
        x = x.to(self._device)

        # Initial convolutional block
        x = self.relu(self.bn1(self.conv1(x)))

        # Residual blocks
        x = self.res_blocks(x)

        # Policy head
        policy = self.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # Value head
        value = self.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def train(self, mode=True):
        """Override train method to handle batch norm layers correctly."""
        super().train(mode)
        if not mode:
            # When setting to eval mode, ensure batch norms are properly configured
            self.bn1.eval()
            self.policy_bn.eval()
            self.value_bn.eval()
        return self

__all__ = ['Connect4Net']