import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from models.nets import MetaMSResNet
from pytorch_msssim import ssim

# --- CONFIG ---
CLUSTERED_DIR = './dataset_clustered'
SAVE_DIR = './results'

# UPDATED: Resumes from your 35k checkpoint to save time
RESUME_CHECKPOINT = './results/trnr_final_35000.pth' 

N, K, R = 12, 1, 5
ALPHA, BETA = 0.001, 0.001
TOTAL_ITER = 50000 
LAMBDA_SSIM = 1.0

# --- 1. TASK SAMPLER WITH SAFETY ---
def sample_trnr_task(device):
    clusters = [d for d in os.listdir(CLUSTERED_DIR) if os.path.isdir(os.path.join(CLUSTERED_DIR, d))]
    selected_clusters = random.sample(clusters, N)
    train_in, train_tar, val_in, val_tar = [], [], [], []

    for cluster in selected_clusters:
        c_path = os.path.join(CLUSTERED_DIR, cluster)
        all_rainy = [f for f in os.listdir(c_path) if f.startswith('rainy_')]
        
        # Guard against empty or sparse clusters
        if len(all_rainy) < 2 * K: 
            continue
            
        pair = random.sample(all_rainy, 2 * K)
        for i, f in enumerate(pair):
            img_rain = np.transpose(cv2.imread(os.path.join(c_path, f)), (2,0,1)).astype(np.float32)/255.0
            img_clean = np.transpose(cv2.imread(os.path.join(c_path, f.replace('rainy_','clean_'))), (2,0,1)).astype(np.float32)/255.0
            
            if i < K:
                train_in.append(img_rain); train_tar.append(img_clean)
            else:
                val_in.append(img_rain); val_tar.append(img_clean)
    
    # Return flag if no clusters were valid
    if not train_in:
        return None

    return {
        'train': (torch.tensor(np.array(train_in)).to(device), torch.tensor(np.array(train_tar)).to(device)),
        'val': (torch.tensor(np.array(val_in)).to(device), torch.tensor(np.array(val_tar)).to(device))
    }

# --- 2. TRAIN LOOP ---
def train_trnr():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MetaMSResNet(in_channels=3, num_filters=48, stages=4).to(device)
    meta_optimizer = optim.Adam(net.parameters(), lr=BETA)
    
    start_iter = 1
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"Loading checkpoint: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        # Resume from 35000 if not specified in checkpoint
        start_iter = checkpoint.get('iteration', 35000) + 1
        print(f"Resuming from iteration {start_iter}")

    def hybrid_loss(pred, target):
        return nn.L1Loss()(pred, target) + LAMBDA_SSIM * (1 - ssim(pred, target, data_range=1.0))

    print(f"TRNR Training targeting {TOTAL_ITER} iterations...")

    for iteration in range(start_iter, TOTAL_ITER + 1):
        meta_optimizer.zero_grad()
        
        for _ in range(R):
            task_data = sample_trnr_task(device)
            
            # --- TASK GUARD: Skips if no valid data found ---
            if task_data is None or task_data['train'][0].shape[0] == 0:
                continue

            # Ensure buffers are not tracking gradients
            for m in net.modules():
                for b in m.buffers():
                    b.requires_grad = False

            # INNER LOOP
            pred = net(task_data['train'][0])
            inner_loss = hybrid_loss(pred, task_data['train'][1])
            
            diff_params = [p for p in net.parameters() if p.requires_grad]
            grads = torch.autograd.grad(inner_loss, diff_params, create_graph=True, allow_unused=True)
            
            old_params = [p.data.clone() for p in net.parameters()]
            grad_idx = 0
            for p in net.parameters():
                if p.requires_grad:
                    if grad_idx < len(grads) and grads[grad_idx] is not None:
                        p.data = p.data - ALPHA * grads[grad_idx]
                    grad_idx += 1

            # OUTER LOOP
            val_pred = net(task_data['val'][0])
            val_loss = hybrid_loss(val_pred, task_data['val'][1])
            
            # Restore weights
            for p, op in zip(net.parameters(), old_params):
                p.data = op
            
            (val_loss / R).backward()

        meta_optimizer.step()

        if iteration % 100 == 0:
            print(f"Iter [{iteration}/{TOTAL_ITER}] Meta-Loss: {val_loss.item():.6f}")

        if iteration % 1000 == 0:
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save({'iteration': iteration, 'model_state_dict': net.state_dict()}, 
                       f'{SAVE_DIR}/trnr_final_{iteration}.pth')

if __name__ == '__main__':
    train_trnr()