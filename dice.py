import os
import numpy as np
import pandas as pd
from skimage import io
from typing import List

# compute dice
def compute_dice_score(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.shape != mask_b.shape:
        raise ValueError("Masks must have the same shape.")

    a = mask_a.flatten().astype(bool)
    b = mask_b.flatten().astype(bool)

    intersection = np.sum(a & b)
    size_a = np.sum(a)
    size_b = np.sum(b)
    
    # Avoid division by zero
    denominator = size_a + size_b
    if denominator == 0:
        return 1.0
    
    dice = (2.0 * intersection) / denominator
    return dice

# compute iou
def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Computes the Jaccard index (IoU) for the positive class (vessels)."""
    if mask_a.shape != mask_b.shape:
        raise ValueError("Masks must have the same shape.")

    a = mask_a.flatten().astype(bool)
    b = mask_b.flatten().astype(bool)

    # Intersection (True Positives)
    intersection = np.sum(a & b)
    
    # Union = (|A| + |B|) - (|A| intersect |B|)
    union = np.sum(a | b)
    
    # Avoid division by zero
    if union == 0:
        return 1.0 # Perfect match for two empty masks

    iou = intersection / union
    return iou

def compute_folder_metrics(ground_truth_folder: str, prediction_folder: str, output_csv_path: str):
    results = []

    gt_files = {f for f in os.listdir(ground_truth_folder) if f.endswith(".png")}
    pred_files = {f for f in os.listdir(prediction_folder) if f.endswith(".png")}
    common_files = sorted(list(gt_files.intersection(pred_files)))

    print(f"Found {len(common_files)} common PNG masks to compare...")
    if not common_files:
        print("No matching PNG files found. Check your folder contents and naming.")
        return

    # Threshold for binarization (pixels < 0.5 are considered vessels/positive class)
    THRESHOLD = 0.5 

    for filename in common_files:
        gt_path = os.path.join(ground_truth_folder, filename)
        pred_path = os.path.join(prediction_folder, filename)

        try:
            gt_mask = io.imread(gt_path, as_gray=True)
            pred_mask = io.imread(pred_path, as_gray=True)
            
            # Binarization: Vessels (Black/low value) are mapped to 1 (True)
            gt_binary = (gt_mask < THRESHOLD).astype(np.uint8)
            pred_binary = (pred_mask < THRESHOLD).astype(np.uint8)

            # Calculate Metrics
            dice_score = compute_dice_score(gt_binary, pred_binary)
            iou_score = compute_iou(gt_binary, pred_binary)
            
            results.append({
                "Filename": filename,
                "Dice_Score": dice_score,
                "IoU_Score": iou_score # 🆕 Added IoU
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append({
                "Filename": filename,
                "Dice_Score": np.nan,
                "IoU_Score": np.nan,
                "Error": str(e)
            })

    # Save results to a DataFrame and CSV
    df = pd.DataFrame(results)
    
    # Calculate overall means
    mean_dice = df["Dice_Score"].mean()
    mean_iou = df["IoU_Score"].mean()
    
    print(f"\n--- Analysis Complete ---")
    print(f"Overall Mean Dice Score: {mean_dice:.4f}")
    print(f"Overall Mean IoU Score: {mean_iou:.4f}")
    
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to: {output_csv_path}")


if __name__ == "__main__":
    # folders
    GROUND_TRUTH_FOLDER = "/Users/aida/Desktop/masks_oleg"  
    PREDICTION_FOLDER = "/Users/aida/Desktop/masks_natalia"      
    OUTPUT_CSV = "/Users/aida/Desktop/dice_score_results_black_vessels.csv"    
    compute_folder_metrics(GROUND_TRUTH_FOLDER, PREDICTION_FOLDER, OUTPUT_CSV)
