import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.nets import MetaMSResNet

# --- CONFIGURATION ---
ITERATIONS = 25000       # Increased to 25,000 for high quality
BATCH_SIZE = 4           # Keep 4 for GTX 1650 stability
LEARNING_RATE = 1e-4     # Standard starting rate
SAVE_INTERVAL = 1000     # Save every 1000 steps
INPUT_DIR = './Rain100L/input'
TARGET_DIR = './Rain100L/target'
SAVE_DIR = './results'

# --- 1. DATASET HANDLER ---
class RainDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])
        self.input_dir = input_dir
        self.target_dir = target_dir

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Read Images
        filename = self.input_files[idx]
        
        # Robust read
        img_in = cv2.imread(os.path.join(self.input_dir, filename))
        img_target = cv2.imread(os.path.join(self.target_dir, filename))

        # Resize to 64x64 patches (Standard for training rain removal)
        # This allows the model to see many examples quickly
        img_in = cv2.resize(img_in, (64, 64))
        img_target = cv2.resize(img_target, (64, 64))

        # Normalize 0-1 and Transpose to (C, H, W)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32) / 255.0
        img_target = np.transpose(img_target, (2, 0, 1)).astype(np.float32) / 255.0

        return torch.from_numpy(img_in), torch.from_numpy(img_target)

# --- 2. TRAINING LOOP ---
def train():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Extended Training on: {device}")
    print(f"Goal: {ITERATIONS} Iterations | Loss: L1 (MAE)")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Initialize Model
    net = MetaMSResNet(in_channels=3, num_filters=32, stages=3)
    net.to(device)
    net.train()

    # CRITICAL: L1 Loss creates sharper images than MSE
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Load Dataset
    dataset = RainDataset(INPUT_DIR, TARGET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    loader_iter = iter(dataloader)

    # Training Loop
    for i in range(1, ITERATIONS + 1):
        try:
            input_tensor, target_tensor = next(loader_iter)
        except StopIteration:
            # Restart the data loader when it runs out of images
            loader_iter = iter(dataloader)
            input_tensor, target_tensor = next(loader_iter)

        input_tensor = input_tensor.to(device)
        target_tensor = target_tensor.to(device)

        # Zero Gradients
        optimizer.zero_grad()

        # Forward Pass
        output = net(input_tensor)

        # Calculate Loss
        loss = criterion(output, target_tensor)

        # Backward Pass
        loss.backward()
        optimizer.step()

        # Logging
        if i % 100 == 0:
            print(f"Iter [{i}/{ITERATIONS}] Loss: {loss.item():.6f}")

        # Save Checkpoint
        if i % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f'checkpoint_{i}.pth')
            torch.save({
                'iteration': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path)
            print(f"--> Checkpoint saved: {save_path}")

    # Save Final
    final_path = os.path.join(SAVE_DIR, 'final_model.pth')
    torch.save(net.state_dict(), final_path)
    print("Training Complete. Final model saved.")

if __name__ == '__main__':
    train()