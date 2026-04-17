# Deep Learning for Single Image Deraining

## Overview
This repository contains an optimized, end-to-end deep learning training pipeline for single image deraining. The project transitions from a fragmented multi-phase training script to a unified, highly efficient approach utilizing a custom **DerainNet** architecture. 

By implementing strict deep learning best practices—such as native-resolution random cropping, perception-driven hybrid loss functions, and automated optimization—this pipeline achieves state-of-the-art level results on the Rain100L dataset, yielding a **+6.67 dB PSNR** improvement over baseline methodologies.

---

## The Challenge
* Rain streaks severely degrade image quality, reducing visibility and altering object appearances. 
* Degraded images negatively impact downstream computer vision tasks, such as object detection, pedestrian tracking, and autonomous driving algorithms. 

Single image deraining is fundamentally an ill-posed inverse problem. A rainy image can be modeled as a linear superimposition:

O = B + R

Where:
* **O** is the observed rainy image.
* **B** is the clean background layer (our target).
* **R** represents the rain streak layer.

**Our Goal:** Train a Convolutional Neural Network (CNN) to accurately estimate R from O, allowing us to retrieve B = O - R.

---

## Key Advancements & Methodology

### 1. Unified Architecture (DerainNet)
Instead of relying on fragmented external files, the proposed approach integrates a robust Residual Network natively. 
* **Feature Extraction:** Initial 3x3 Convolution with PReLU activation.
* **Deep Residual Learning:** 5 identical Residual Blocks. It is easier for the network to learn the sparse difference (the rain streaks R) than to map directly to the complex background image (B).
* **Output Layer:** Reconstructs the rain mask. The final clean image is computed precisely via B = O - R, constrained between [0.0, 1.0].

### 2. Native Resolution Patching (128x128)
* **The Flaw of Resizing:** The baseline approach used cv2.resize() to compress full HD images into 64x64 tensors. Because rain streaks are high-frequency signals, downsampling acts as a low-pass filter, blurring and destroying the structural information the model needs to learn.
* **Our Solution:** We extract random 128x128 spatial patches directly from the original, uncompressed input images. This preserves exact 1:1 pixel data and high-frequency rain streak edges, and inherently acts as a form of spatial data augmentation.

### 3. Perception-Driven Hybrid Loss
L1 Loss (Mean Absolute Error) only enforces global color correctness and ignores spatial relationships, frequently resulting in visually blurry backgrounds. We introduce a combined loss function leveraging the Structural Similarity Index Measure (SSIM):

L_total = alpha(1 - SSIM(y_hat, y)) + (1 - alpha)||y_hat - y||_1 

*The SSIM component penalizes the network for destroying structural information, forcing the reconstruction of sharp edges and textures (alpha = 0.84).*

### 4. Optimization & Regularization Strategies
* **Cosine Annealing (Automated):** Replaces clunky manual checkpoint loading. The learning rate smoothly decays following a cosine curve over 35,000 iterations. This allows rapid convergence early on, and ultra-fine precision updates near the end to prevent gradient oscillation.
* **Rigorous Data Augmentation:** The proposed method applies stochastic 50% Horizontal and Vertical flips from Iteration 1. This exponentially increases the geometric diversity of rain angles.

---

## Dataset & Training Setup
* **Dataset:** Rain100L (Synthetic rain dataset) 
  * Training Set: 200 image pairs (Rainy/Clean) 
  * Testing Set: 100 image pairs for evaluation 
* **Hardware:** NVIDIA GTX 1650 (or equivalent CUDA-enabled GPU) 
* **Optimizer:** Adam (beta_1 = 0.9, beta_2 = 0.999) 
* **Batch Size:** 4 
* **Iterations:** 35,000 Total 

---

## Quantitative Results on Rain100L 

The proposed pipeline drastically outperforms the baseline methodology across standard evaluation metrics:

| Methodology | PSNR (dB) ↑ | SSIM ↑ |
| :--- | :--- | :--- |
| **Baseline Script** (Resized, L1) | 28.45 | 0.8612 |
| **Proposed Script** (Cropped, L1+SSIM) | **35.12** | **0.9654** |
| **Net Improvement** | **+6.67 dB** | **+0.1042** |

*Note: A PSNR increase of +6dB mathematically signifies that the mean square error has been reduced by roughly a factor of 4.*

### Ablation Study: Contribution of Components 
To prove the efficacy of our pipeline upgrades, we evaluated the individual contribution of each component incrementally.

| Base Net | Crop (Not Resize) | SSIM Loss | Augmentation | PSNR | SSIM |
| :---: | :---: | :---: | :---: | :---: | :---: |
| X | | | | 28.45 | 0.8612 |
| X | X | | | 31.80 | 0.9105 |
| X | X | X | | 34.05 | 0.9520 |
| X | X | X | X | **35.12** | **0.9654** |

*Insight: Moving from squishing the image to patching the image yielded the single highest jump in performance (+3.35 dB).*

---
## Qualitative Visual Results

The unified DerainNet pipeline effectively removes dense rain streaks across highly varied background textures while successfully preserving high-frequency edge details and accurate color saturation.

### Arctic Expedition
![Arctic Expedition Comparison](assets/Screenshot_2026-03-27_122808.jpg)

### Wildlife (Deer)
![Deer Comparison](assets/Screenshot_2026-03-27_122816.jpg)

### Underwater Marine Life
![Fish Comparison](assets/Screenshot_2026-03-27_122824.jpg)

### Horses in Field
![Horses Comparison](assets/Screenshot_2026-03-27_122840.jpg)

## Usage & Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/DerainNet-Unified-Pipeline.git](https://github.com/yourusername/DerainNet-Unified-Pipeline.git)
cd DerainNet-Unified-Pipeline

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --dataset path/to/Rain100L --batch_size 4

# Run inference
python test.py --input path/to/test_images --weights best_model.pth
