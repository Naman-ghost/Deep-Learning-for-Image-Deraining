import cv2
import numpy as np
import math
import os
import tqdm
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def get_scores():
    # --- CONFIG ---
    CLEANED_DIR = './cleaned_results'
    TARGET_DIR = './Rain100L/target'
    
    # Get all cleaned files
    cleaned_files = [f for f in os.listdir(CLEANED_DIR) if f.startswith('cleaned_')]
    
    if not cleaned_files:
        print(f"Error: No images found in {CLEANED_DIR}. Did you run process_all.py?")
        return

    total_psnr = 0
    total_ssim = 0
    count = 0

    print(f"Evaluating 50,000 Iteration TRNR Results for {len(cleaned_files)} images...")

    for filename in tqdm.tqdm(cleaned_files):
        # Match cleaned_X.png with X.png
        original_name = filename.replace('cleaned_', '')
        
        cleaned_path = os.path.join(CLEANED_DIR, filename)
        target_path = os.path.join(TARGET_DIR, original_name)

        img_cleaned = cv2.imread(cleaned_path)
        img_target = cv2.imread(target_path)

        if img_cleaned is None or img_target is None:
            continue

        # PSNR
        psnr_val = calculate_psnr(img_cleaned, img_target)
        
        # SSIM - using channel_axis=2 for color images
        ssim_val = ssim(img_cleaned, img_target, channel_axis=2, data_range=255)

        total_psnr += psnr_val
        total_ssim += ssim_val
        count += 1

    if count > 0:
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        
        print(f"\n" + "="*50)
        print(f"--- TRNR FINAL QUANTITATIVE RESULTS (50,000 Iterations) ---")
        print(f"Total Images Processed: {count}")
        print(f"Average PSNR:           {avg_psnr:.2f} dB")
        print(f"Average SSIM:           {avg_ssim:.4f}")
        print(f"Paper Target PSNR:      38.17 dB")
        print(f"Paper Target SSIM:      0.9814")
        print("="*50)
    else:
        print("Error: Could not calculate metrics. Check file naming.")

if __name__ == "__main__":
    get_scores()