import torch
import cv2
import numpy as np
import os
import tqdm
from models.nets import MetaMSResNet

# --- CONFIG ---
# UPDATED: Pointing to your final 50,000 iteration checkpoint
CHECKPOINT_PATH = './results/trnr_final_50000.pth' 
INPUT_DIR = './Rain100L/input'
OUTPUT_DIR = './cleaned_results'

def process_batch():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Matches your high-capacity TRNR architecture
    net = MetaMSResNet(in_channels=3, num_filters=48, stages=4)
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Load the final meta-trained weights
    # Note: weights_only=False is used here to ensure compatibility with saved dicts
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device).eval()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.png')])

    print(f"Generating Final TRNR results (50k iterations) for {len(files)} images...")

    for filename in tqdm.tqdm(files):
        img = cv2.imread(os.path.join(INPUT_DIR, filename))
        if img is None: continue
        
        # Preprocess: (H, W, C) -> (C, H, W) and scale to [0, 1]
        input_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0) / 255.0
        input_tensor = input_tensor.to(device)

        # Inference: The model predicts the background 'x' by subtracting rain 'n'
        with torch.no_grad():
            output = net(input_tensor)

        # Post-process: Convert back, rescale, and clip to valid pixel range
        model_output = output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
        final_img = np.clip(model_output, 0, 255).astype(np.uint8)
        
        # Saving with 'cleaned_' prefix for the metrics script to pick up
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"cleaned_{filename}"), final_img)

    print(f"\n[DONE] All 50k results saved in: {OUTPUT_DIR}")

if __name__ == '__main__':
    process_batch()