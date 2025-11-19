# apm_kidney/main.py
import os
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("TkAgg")  # if you need Tk on your machine
from readAndOpenImage import openFile, openFolder
from smoothImage import smoothBSpline
from postProcessing import morphologicalOps, binarize
from normalizeImage import normalizeWithPercentile, normalizeWithThreshold
from PIL import Image
import numpy as  np
import imageio.v3 as iio   # pip install imageio
import tifffile as tiff

def run_one(path):
    # 1. Open image
    ds, da, px_um = openFile(path)

    # 2. Cut bckg and artifacts with threshold, then normalize to [0, 1] range
    imgNorm = normalizeWithThreshold(da,  thr = 0.148, plot = False)
    
    # 3. Smooth image (B-spline)
    img_bspline = smoothBSpline(imgNorm, plot = False)
    
    # 4. Normalize with percentiles again after smoothing to get the [0, 1] range, also handle outliers
    imgNormP = normalizeWithPercentile(img_bspline, p_lo=1, p_hi=99, plot=False)

    # Clip to [0, 0.25] range to enhance vessel contrast
    imgNormP = np.clip(imgNormP, a_min = 0, a_max = 0.25)
    imgNormP = (imgNormP - imgNormP.min()) / (imgNormP.max() - imgNormP.min())

    # 5. Binarize to get vessel mask
    mask = binarize(imgNormP, thr = 0.2, plot = False)
    
    # 6. Apply morphological operations to clean up the vessel mask
    mask_clean = morphologicalOps(mask, ds["tlsc_tmean"].values, px_um = px_um, 
                                 plot = False, min_obj_area_um2=180.0)

    return mask_clean


def _to_bool(mask):
    # Whatever run_one returns, force strict binary in memory
    return (mask > 0)

def _save_mask_lossless(mask_bool, fname_noext, out_dir=None, as_tiff=False):
    os.makedirs(out_dir or ".", exist_ok=True)
    outbase = os.path.join(out_dir or ".", f"mask_{fname_noext}")
    if as_tiff:
        # True 1-bit per pixel TIFF
        import tifffile as tiff
        tiff.imwrite(f"{outbase}.tif", mask_bool.astype(bool))
    else:
        # Universal: PNG with 0/255
        iio.imwrite(f"{outbase}.png", (mask_bool.astype(np.uint8) * 255))

def runAll(path, cols=4, show=True, cmap_img="gray", overlay_alpha=0.35, out_dir="masks"):
    """
    Run segmentation on all .nc files in the given folder path, 
    display results in 3 figures: originals, masks, overlays.
    Parameters:
        path (str): Folder path containing .nc files.
        cols (int): Number of columns in the display grid.
        show (bool): Whether to show the plots.
        cmap_img (str): Colormap for original images.
        overlay_alpha (float): Alpha for mask overlay.
    """

    # 1) Get file names using existing function
    image_files = openFolder(path, plot=False)
    if not image_files:
        print("No .nc files found in:", path); return

    originals, masks, titles = [], [], []

    # 2) Run per file
    for fname in image_files:
        fullpath = os.path.join(path, fname)
        stem = os.path.splitext(fname)[0]
        try:
            _, original, _ = openFile(fullpath)
            mask = run_one(fullpath)
            mask_bool = _to_bool(mask)

            originals.append(original)
            masks.append(mask_bool)
            titles.append(fname)

            # SAVE — lossless, strictly binary
            #_save_mask_lossless(mask_bool, stem, out_dir=out_dir, as_tiff=False)  # PNG
            # If your teammates use ImageJ/Fiji a lot, consider as_tiff=True instead.

        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")

    if not originals or not masks:
        print("Nothing to plot."); return

    n = len(originals)
    cols = max(1, int(cols))
    rows = (n + cols - 1) // cols

    # -------- Figure 1: Originals --------
    fig1, axes1 = plt.subplots(rows, cols, figsize=(4.2*cols, 4.2*rows), constrained_layout=True)
    axes1 = np.atleast_1d(axes1).ravel()
    for i, ax in enumerate(axes1):
        if i < n:
            ax.imshow(originals[i], cmap=cmap_img, origin="lower")
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    # -------- Figure 2: Masks (force nearest + binary range) --------
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4.2*cols, 4.2*rows), constrained_layout=True)
    axes2 = np.atleast_1d(axes2).ravel()
    for i, ax in enumerate(axes2):
        if i < n:
            ax.imshow(masks[i], cmap="gray", origin="lower",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    # -------- Figure 3: Overlays (mask tinted, nearest) --------
    fig3, axes3 = plt.subplots(rows, cols, figsize=(4.2*cols, 4.2*rows), constrained_layout=True)
    axes3 = np.atleast_1d(axes3).ravel()
    for i, ax in enumerate(axes3):
        if i < n:
            ax.imshow(originals[i], cmap=cmap_img, origin="lower")
            m = np.ma.masked_where(~masks[i], masks[i])
            ax.imshow(m, cmap="autumn", origin="lower",
                      interpolation="nearest", alpha=overlay_alpha, vmin=0, vmax=1)
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    if show:
        plt.show()

def runAndSaveImgs(path):
    """
    Run segmentation on all .nc files in the given folder path,
    save resulting images
    """
    
    # 1) Get file names using existing function
    image_files = openFolder(path, plot=False)   # returns just filenames

    if not image_files:
        print("No .nc files found in:", path)
        return

    originals = []
    masks = []
    titles = []

    # 2) Run per file
    for fname in image_files:
        fullpath = os.path.join(path, fname)
        try:
            _, original, _ = openFile(fullpath) 
            mask = run_one(fullpath)   # your existing function

            originals.append(original)
            masks.append(mask)
            titles.append(fname)
            
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")

    pass

if __name__ == "__main__":
    # --- Path to your file ---
    path = Path('C:/Users/natal/Documents/APM_Kidney_Project/')
    #filename = path / 'images/fromOleg/20241023b_diabet_f_vehicle_adult.nc' #thr = 0.148
    #filename = path / 'images/fromOleg/20241028a_diabet_m_vehicle_adult.nc' #thr = 0.143
    filename = path / 'images/fromOleg/20241011b_control_f_vehicle_adult.nc'   #thr = 0.1334
    
    #run_one(str(filename))
    
    # Optionally see all:
    folder = r"C:/Users/natal/Documents/APM_Kidney_Project/images/fromOleg"
    runAll(path=folder, cols=4, show=True, 
           cmap_img="gray", overlay_alpha=0.35)