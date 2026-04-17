import os
import cv2
import numpy as np
import tqdm
import shutil

# --- CONFIG ---
INPUT_DIR = './Rain100L/input'
TARGET_DIR = './Rain100L/target'
OUTPUT_DIR = './dataset_clustered'
PATCH_SIZE = 64
STRIDE = 32 # Higher overlap for more patches
# Adjust THRESHOLD to control cluster count. 
# Lower value = more strict = more clusters (target ~4000) [cite: 409, 554]
THRESHOLD = 0.85 

def get_patch_feature(patch):
    """Extracts color and texture features as per Algorithm 1 [cite: 244]"""
    # Calculate color histogram
    hist = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

def run_clustering():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    image_files = sorted(os.listdir(INPUT_DIR))
    clusters = [] # Stores feature vectors of cluster centers [cite: 285]

    print("Generating patches and clustering...")
    for filename in tqdm.tqdm(image_files):
        img_in = cv2.imread(os.path.join(INPUT_DIR, filename))
        img_tar = cv2.imread(os.path.join(TARGET_DIR, filename.replace('input', 'target'))) # adjust as needed

        h, w, _ = img_in.shape
        for y in range(0, h - PATCH_SIZE + 1, STRIDE):
            for x in range(0, w - PATCH_SIZE + 1, STRIDE):
                patch_in = img_in[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                patch_tar = img_tar[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                
                feat = get_patch_feature(patch_tar)
                
                # Match against existing clusters [cite: 285]
                matched_idx = -1
                for i, center_feat in enumerate(clusters):
                    score = cv2.compareHist(feat, center_feat, cv2.HISTCMP_CORREL)
                    if score > THRESHOLD:
                        matched_idx = i
                        break
                
                if matched_idx == -1:
                    # Create new cluster [cite: 285]
                    matched_idx = len(clusters)
                    clusters.append(feat)
                
                # Save patch pair to cluster folder
                cluster_path = os.path.join(OUTPUT_DIR, f"cluster_{matched_idx}")
                os.makedirs(cluster_path, exist_ok=True)
                patch_id = f"{filename}_{y}_{x}.png"
                cv2.imwrite(os.path.join(cluster_path, f"rainy_{patch_id}"), patch_in)
                cv2.imwrite(os.path.join(cluster_path, f"clean_{patch_id}"), patch_tar)

    print(f"Clustering complete. Created {len(clusters)} clusters.")

if __name__ == '__main__':
    run_clustering()