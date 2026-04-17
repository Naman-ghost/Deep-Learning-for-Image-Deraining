import torch
import cv2
import numpy as np
import os
from models.nets import MetaMSResNet

# --- CONFIGURATION ---
DEFAULT_CHECKPOINT = './results/checkpoint_25000.pth'
INPUT_IMAGE = './Rain100L/input/1.png'
OUTPUT_IMAGE = 'real_result_cleaned.png'

def color_fix_and_denoise(cleaned, original):
    cleaned = cleaned.astype(np.float32)
    original = original.astype(np.float32)
    cleaned = np.clip(cleaned, 0, 255)
    for i in range(3): 
        mean_clean = cleaned[:,:,i].mean() + 1e-6
        mean_orig = original[:,:,i].mean()
        ratio = mean_orig / mean_clean
        cleaned[:,:,i] = cleaned[:,:,i] * ratio
    return np.clip(cleaned, 0, 255).astype(np.uint8)

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # --- FIX: Proper path resolution ---
    checkpoint_to_use = DEFAULT_CHECKPOINT
    
    if not os.path.exists(checkpoint_to_use):
        print(f"Target {checkpoint_to_use} not found. Searching for latest...")
        if os.path.exists('./results/'):
            checkpoints = [f for f in os.listdir('./results/') if f.startswith('checkpoint_') and f.endswith('.pth')]
            if checkpoints:
                checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
                checkpoint_to_use = os.path.join('./results', checkpoints[0])
                print(f"Found latest: {checkpoint_to_use}")
            else:
                print("Error: No checkpoints found in ./results/. Please train first.")
                return
        else:
            print("Error: ./results/ folder does not exist.")
            return

    # Load Model
    net = MetaMSResNet(in_channels=3, num_filters=32, stages=3)
    print(f"Loading weights from {checkpoint_to_use}...")
    checkpoint = torch.load(checkpoint_to_use, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device).eval()

    # Load Image
    img_path = INPUT_IMAGE
    if not os.path.exists(img_path):
        img_path = img_path.replace('.png', '.jpg')
    
    if not os.path.exists(img_path):
        print(f"Error: Could not find image at {INPUT_IMAGE}")
        return

    print(f"Processing: {img_path}")
    img = cv2.imread(img_path)
    input_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = net(input_tensor)

    output_img = output.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0
    final_result = color_fix_and_denoise(output_img, img)

    cv2.imwrite(OUTPUT_IMAGE, final_result)
    print(f"Success! Optimized result saved as '{OUTPUT_IMAGE}'")

if __name__ == '__main__':
    test()