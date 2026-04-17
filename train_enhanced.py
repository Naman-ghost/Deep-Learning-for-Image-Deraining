import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from models.nets import MetaMSResNet

# --- CONFIGURATION ---
START_ITER = 25000       # Resuming from your current best
TOTAL_ITER = 35000       # 10,000 more steps of fine-tuning
BATCH_SIZE = 4           
FINE_LR = 5e-5           # Lower learning rate for precision
SAVE_INTERVAL = 1000     
INPUT_DIR = './Rain100L/input'
TARGET_DIR = './Rain100L/target'
SAVE_DIR = './results'
CHECKPOINT_PATH = './results/checkpoint_25000.pth'

# --- 1. DATASET WITH AUGMENTATION ---
class RainDataset(Dataset):
    def __init__(self, input_dir, target_dir):
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])
        self.input_dir = input_dir
        self.target_dir = target_dir

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        filename = self.input_files[idx]
        img_in = cv2.imread(os.path.join(self.input_dir, filename))
        img_target = cv2.imread(os.path.join(self.target_dir, filename))

        img_in = cv2.resize(img_in, (64, 64))
        img_target = cv2.resize(img_target, (64, 64))

        # ML CONCEPT: Data Augmentation (Flips & Rotations)
        # Helps the model generalize instead of just memorizing
        if random.random() > 0.5:
            img_in = cv2.flip(img_in, 1) # Horizontal Flip
            img_target = cv2.flip(img_target, 1)

        # Normalize and Transpose
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32) / 255.0
        img_target = np.transpose(img_target, (2, 0, 1)).astype(np.float32) / 255.0

        return torch.from_numpy(img_in), torch.from_numpy(img_target)

# --- 2. TRAINING LOOP ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize Model
    net = MetaMSResNet(in_channels=3, num_filters=32, stages=3)
    net.to(device)

    # Load Existing Checkpoint
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint: {CHECKPOINT_PATH} for fine-tuning...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_at = checkpoint['iteration']
    else:
        print("Error: 25k checkpoint not found. Run standard training first.")
        return

    net.train()
    criterion = nn.L1Loss() 
    
    # Use the lower learning rate for finer weight adjustments
    optimizer = optim.Adam(net.parameters(), lr=FINE_LR)

    dataset = RainDataset(INPUT_DIR, TARGET_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    loader_iter = iter(dataloader)

    print(f"Fine-Tuning from Iteration {start_at} to {TOTAL_ITER}...")

    for i in range(start_at + 1, TOTAL_ITER + 1):
        try:
            input_tensor, target_tensor = next(loader_iter)
        except StopIteration:
            loader_iter = iter(dataloader)
            input_tensor, target_tensor = next(loader_iter)

        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

        optimizer.zero_grad()
        output = net(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iter [{i}/{TOTAL_ITER}] Loss: {loss.item():.6f}")

        if i % SAVE_INTERVAL == 0:
            save_path = os.path.join(SAVE_DIR, f'checkpoint_{i}.pth')
            torch.save({
                'iteration': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, save_path)
            print(f"--> Enhanced Checkpoint saved: {save_path}")

    print("Enhanced Training Complete!")

if __name__ == '__main__':
    train()