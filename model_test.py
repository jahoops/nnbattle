import torch
from nnbattle.agents.alphazero.network import Connect4Net

# Define the model architecture with state_dim=3
model = Connect4Net(state_dim=3, action_dim=7)  # Changed from 2 to 3

# Save the model state_dict
MODEL_PATH = "mnt/ramdisk/alphazero_model_final.pth"
torch.save(model.state_dict(), MODEL_PATH)
print("Model saved successfully.")