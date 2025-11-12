import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import xarray as xr
from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    estimate_sigma,
    denoise_tv_chambolle,
)
from skimage.filters import sato
from skimage.morphology import remove_small_objects, binary_erosion
import os
from skimage.measure import label, regionprops, perimeter as sk_perimeter
import pandas as pd

def plot_processing_stages(original, normalized, denoised, enhanced, mask, df_stats=None):
    """Plot each processing stage and optional quality assessment."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ax = axes.ravel()

    ax[0].imshow(original, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(normalized, cmap="gray")
    ax[1].set_title("After Normalization")
    ax[1].axis("off")

    ax[2].imshow(denoised, cmap="gray")
    ax[2].set_title("After Noise Reduction")
    ax[2].axis("off")

    ax[3].imshow(enhanced, cmap="gray")
    ax[3].set_title("After Enhancement")
    ax[3].axis("off")

    ax[4].imshow(mask, cmap="gray")
    ax[4].set_title("Final Vessel Mask")
    ax[4].axis("off")

    if df_stats is not None and "FractionVessel" in df_stats.columns:
        ax[5].bar(df_stats["Box"], df_stats["FractionVessel"], color="tomato")
        ax[5].set_title("Quality: Vessel Fraction per Box")
        ax[5].set_xlabel("Box")
        ax[5].set_ylabel("Fraction (%)")
    else:
        ax[5].axis("off")

    plt.tight_layout()
    plt.show()


# STEP 1: Open dataset and normalize
def preprocess_image(filename):
    ds = xr.open_dataset(filename)
    da = ds["tlsc_tmean"].astype("float32")
    da_np = da.values

    thr = np.nanpercentile(da_np, 98)
    da_clip = np.clip(da_np, float(da_np.min()), float(thr))
    lo, hi = np.nanpercentile(da_clip, [1, 99])
    da_norm = (da_clip - lo) / (hi - lo)
    da_norm = np.clip(da_norm, 0, 1)
    plt.imshow(da_norm)
    return da_norm


# STEP 2: Smooth image
def smooth_image(img):
    img_bilat = denoise_bilateral(img, sigma_color=0.05, sigma_spatial=2, channel_axis=None)
    sigma_est = np.mean(estimate_sigma(img, channel_axis=None))
    patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=None)
    img_nlm = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, **patch_kw)
    img_tv = denoise_tv_chambolle(img, weight=0.05)
    return img_tv


# STEP 3: Vessel enhancement and mask cleanup
def extract_vessels(img):
    img_inv = 1 - img
    sigmas = np.linspace(1, 6, 8)
    vesselness = sato(img_inv, sigmas=sigmas, black_ridges=False)
    vesselness = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min())
    t = np.percentile(vesselness, 94)
    mask = vesselness >= t
    mask_clean = binary_erosion(mask)
    mask_clean = remove_small_objects(mask_clean, min_size=250)
    return mask_clean


# STEP 4: Interactive draggable boxes
class DraggableRectangle:
    def __init__(self, rect):
        self.rect = rect
        self.press = None
        self.connect()

    def connect(self):
        self.cidpress = self.rect.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.rect.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.rect.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.rect.axes: 
            return
        contains, _ = self.rect.contains(event)
        if not contains: 
            return
        x0, y0 = self.rect.xy
        self.press = x0, y0, event.xdata, event.ydata

    def on_motion(self, event):
        if self.press is None or event.inaxes != self.rect.axes: 
            return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.rect.set_xy((x0 + dx, y0 + dy))
        self.rect.figure.canvas.draw()

    def on_release(self, event):
        self.press = None
        self.rect.figure.canvas.draw()


def boxes_from_clicks(img, mask, box_size=256, num_boxes=5):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray', alpha=0.6)
    plt.title(f"Click {num_boxes} points to select box centers")
    plt.axis('off')
    plt.tight_layout()
    clicks = plt.ginput(num_boxes, timeout=-1)
    plt.close()

    boxes = []
    rows, cols = mask.shape
    half = box_size // 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap='gray', alpha=0.6)
    ax.imshow(mask, cmap='Reds', alpha=0.3)
    ax.set_title("Adjust boxes (drag to move), close window when done")
    draggable_boxes = []

    for cx, cy in clicks:
        cy, cx = int(cy), int(cx)
        minr = max(0, cy - half)
        maxr = min(rows, cy + half)
        minc = max(0, cx - half)
        maxc = min(cols, cx + half)
        rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                 linewidth=2, edgecolor="lime", facecolor="none")
        ax.add_patch(rect)
        draggable_boxes.append(DraggableRectangle(rect))
        boxes.append((minr, minc, maxr, maxc))

    plt.show()

    # Update boxes after dragging
    final_boxes = []
    for rect in draggable_boxes:
        x, y = rect.rect.get_xy()
        w = rect.rect.get_width()
        h = rect.rect.get_height()
        final_boxes.append((int(y), int(x), int(y + h), int(x + w)))

    return final_boxes


# STEP 5: Save boxes and corresponding masks
def save_boxes_and_masks(img, mask, boxes, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    for i, (minr, minc, maxr, maxc) in enumerate(boxes):
        box_img = img[minr:maxr, minc:maxc]
        mask_img = mask[minr:maxr, minc:maxc]
        img_filename = os.path.join(output_dir, f"box_{i+1}_img.png")
        mask_filename = os.path.join(output_dir, f"box_{i+1}_mask.png")
        plt.imsave(img_filename, box_img, cmap="gray")
        plt.imsave(mask_filename, mask_img, cmap="gray")
        saved_files.append((img_filename, mask_filename))
    return saved_files


# STEP 6: Zoom on selected box
def zoom_box(mask, boxes):
    plt.figure(figsize=(8, 8))
    plt.imshow(mask, cmap='gray')
    plt.title("Click on a box to zoom (mask only)")
    for (minr, minc, maxr, maxc) in boxes:
        rect = patches.Rectangle((minc, minr), maxc-minc, maxr-minr,
                                 linewidth=2, edgecolor="lime", facecolor="none")
        plt.gca().add_patch(rect)
    plt.axis("off")
    plt.tight_layout()
    click = plt.ginput(1, timeout=-1)
    plt.close()

    if click:
        x_click, y_click = click[0]
        for idx, (minr, minc, maxr, maxc) in enumerate(boxes):
            if minc <= x_click <= maxc and minr <= y_click <= maxr:
                mask_zoom = mask[minr:maxr, minc:maxc]
                plt.figure(figsize=(6, 6))
                plt.imshow(mask_zoom, cmap='gray')
                plt.title(f"Zoomed Mask Box {idx+1}")
                plt.axis("off")
                plt.show()
                return

    print("Click was outside any box.")


# STEP 7: Quantify vessels in saved boxes (added mean radius)
def quantify_boxes(saved_files, output_dir):
    stats = []
    for i, (img_file, mask_file) in enumerate(saved_files):
        mask_img = plt.imread(mask_file)
        plt.imshow(mask_img)

        if mask_img.ndim == 3:
            mask_img = mask_img[..., 0]

        mask_img = mask_img > 0.5
        labeled = label(mask_img)
        props = regionprops(labeled)

        total_area = mask_img.size
        vessel_area = mask_img.sum()
        vessel_fraction = vessel_area / total_area * 100
        n_objects = len(props)
        total_perimeter = sk_perimeter(mask_img, neighborhood=3)

        # Calculate equivalent radius for each vessel: r = sqrt(area / pi)
        radii = [np.sqrt(p.area / np.pi) for p in props if p.area > 0]
        mean_radius = np.mean(radii) if len(radii) > 0 else 0

        stats.append({
            "Box": i+1,
            "TotalPixels": total_area,
            "VesselPixels": vessel_area,
            "FractionVessel": vessel_fraction,
            "NumObjects": n_objects,
            "TotalPerimeter": total_perimeter,
            "MeanRadius": mean_radius,  # new statistic added
        })

    df = pd.DataFrame(stats)
    csv_path = os.path.join(output_dir, "box_statistics.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved statistics: {csv_path}")
    return df


def plot_vessel_fraction(df, output_dir):
    plt.figure(figsize=(6,4))
    plt.bar(df["Box"], df["FractionVessel"], color="tomato")
    plt.xlabel("Box")
    plt.ylabel("% Vessel Area")
    plt.title("Vessel Fraction per Box")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "vessel_fraction_barplot.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"Saved plot: {plot_path}")


# MAIN
if __name__ == "__main__":
    filename = "/Users/aida/Desktop/APM/time_averaged_tLSC_and_vessel_masks/20241029c_control_f_vehicle_adult.nc"
    output_dir = "/Users/aida/Desktop/APM/SelectedBoxes"

    img_norm = preprocess_image(filename)
    ds = xr.open_dataset(filename)
    img_raw = ds["tlsc_tmean"].astype("float32").values
    img_smooth = smooth_image(img_norm)
    mask = extract_vessels(img_smooth)
    boxes = boxes_from_clicks(img_norm, mask, box_size=1024, num_boxes=1)
    saved_files = save_boxes_and_masks(img_norm, mask, boxes, output_dir)
    zoom_box(mask, boxes)
    df_stats = quantify_boxes(saved_files, output_dir)
    print(df_stats)
    plot_processing_stages(img_raw, img_norm, img_smooth, 1 - img_smooth, mask, df_stats)
