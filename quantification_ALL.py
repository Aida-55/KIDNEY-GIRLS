#CODE FOR QUANTIFICATION OF SEGMENTATIONS
#import libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from math import ceil
from skimage.measure import label, regionprops, profile_line, perimeter as sk_perimeter
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.stats import ttest_ind

#obtain data and organize it
def parse_metadata_from_filename(filename_stem):
    name = filename_stem.lower()

    sample_id = name

    if "diabet" in name:
        condition = "diabetic"
    elif "control" in name:
        condition = "control"
    else:
        condition = "unknown"

    if "_m_" in name:
        sex = "male"
    elif "_f_" in name:
        sex = "female"
    else:
        sex = "unknown"

    return {"SampleID": sample_id, "Condition": condition, "Sex": sex}

#calculate diameter using FWHM
def fwhm_trough(profile, edge_n=5):
    n = len(profile)
    if n < edge_n * 2 + 3:
        return None

    left_base = np.mean(profile[:edge_n])
    right_base = np.mean(profile[-edge_n:])
    baseline = 0.5 * (left_base + right_base)

    i_min = int(np.argmin(profile))
    trough = profile[i_min]
    depth = baseline - trough
    if depth <= 1e-8:
        return None

    half = baseline - 0.5 * depth

    l = None
    for k in range(i_min, 0, -1):
        if profile[k] > half and profile[k - 1] <= half:
            denom = (profile[k] - profile[k - 1]) + 1e-12
            frac = (half - profile[k - 1]) / denom
            l = (k - 1) + frac
            break

    r = None
    for k in range(i_min, n - 1):
        if profile[k] <= half and profile[k + 1] > half:
            denom = (profile[k + 1] - profile[k]) + 1e-12
            frac = (half - profile[k]) / denom
            r = k + frac
            break

    if l is None or r is None or r <= l:
        return None
    return r - l


def measure_diameter_fwhm(raw_img, mask_bool, pixel_size_um, step_px=3, half_width_px=20, smooth_um=20.0):
    skel = skeletonize(mask_bool)
    if not np.any(skel):
        return pd.DataFrame(columns=["VesselLabel", "MeanDiameter_px_fwhm", "MeanDiameter_um_fwhm", "NumSamples_fwhm"])

    dist = distance_transform_edt(mask_bool)
    gy, gx = np.gradient(dist.astype(np.float32))
    grad_norm = np.sqrt(gx**2 + gy**2) + 1e-12
    nx = gx / grad_norm
    ny = gy / grad_norm

    labeled = label(mask_bool, connectivity=2)
    sigma_px = (smooth_um / pixel_size_um) / np.sqrt(8.0 * np.log(2.0))

    rows = []
    for p in regionprops(labeled):
        coords = np.argwhere((labeled == p.label) & skel)
        if coords.size == 0:
            continue

        coords = coords[::max(1, step_px)]
        local_diams = []

        for y, x in coords:
            n_y = float(ny[y, x])
            n_x = float(nx[y, x])
            norm = np.hypot(n_x, n_y) + 1e-12
            n_y /= norm
            n_x /= norm

            y0, x0 = y - n_y * half_width_px, x - n_x * half_width_px
            y1, x1 = y + n_y * half_width_px, x + n_x * half_width_px

            prof = profile_line(raw_img.astype(np.float32), (y0, x0), (y1, x1),
                                mode="reflect", order=1)
            w = fwhm_trough(prof, edge_n=5)
            if w is not None:
                local_diams.append(w)

        if len(local_diams) == 0:
            continue

        local_diams = gaussian_filter1d(local_diams, sigma=max(0.5, sigma_px))
        mean_px = float(np.mean(local_diams))
        mean_um = mean_px * pixel_size_um
        rows.append((int(p.label), mean_px, mean_um, len(local_diams)))

    return pd.DataFrame(rows, columns=["VesselLabel", "MeanDiameter_px_fwhm", "MeanDiameter_um_fwhm", "NumSamples_fwhm"])

#save mask as png
def save_mask_as_png(mask, output_filepath):
    """
    Saves a binary mask as a grayscale PNG image with
    vessels in black and background in white.
    """
    # 1. Invert the mask: Vessels (1/True) become 0, Background (0/False) becomes 1.
    # The negation operator (~) works for boolean arrays in numpy.
    inverted_mask = ~mask
    
    # 2. Convert to uint8 (0 or 1) for saving
    img_data = inverted_mask.astype(np.uint8)

    # 3. Save: cmap="gray" maps 0 to black and 1 to white.
    # Since we inverted the mask: Vessels (0) -> Black; Background (1) -> White.
    plt.imsave(output_filepath, img_data, cmap="gray", vmin=0, vmax=1)

#quantify nc file
def quantify_nc_file(nc_path, pixel_size_um=1.0, raw_var="tlsc_tmean", mask_var="vessel_mask", output_dir=None):
    ds = xr.open_dataset(nc_path)

    raw = ds[raw_var].values.astype(np.float32)
    mask = ds[mask_var].values.astype(np.float32) > 0.5

    stem = os.path.splitext(os.path.basename(nc_path))[0]
    meta = parse_metadata_from_filename(stem)


    if output_dir is not None:
        mask_filename = os.path.join(output_dir, f"{stem}.png") # Added BW for clarity
        save_mask_as_png(mask, mask_filename) # save_mask_as_png handles the inversion


    fraction_percent = 100.0 * float(mask.sum()) / float(mask.size)
    labeled = label(mask, connectivity=2)
    num_vessels = len(regionprops(labeled))
    total_perimeter_px = float(sk_perimeter(mask, neighborhood=3))

    df_fwhm = measure_diameter_fwhm(raw, mask, pixel_size_um)

    mean_fwhm_um = float(df_fwhm["MeanDiameter_um_fwhm"].mean()) if len(df_fwhm) else 0.0

    summary = {
        "SampleID": meta["SampleID"],
        "Condition": meta["Condition"],
        "Sex": meta["Sex"],
        "File": os.path.basename(nc_path),
        "FractionVessel_percent": fraction_percent,
        "NumVessels": num_vessels,
        "TotalPerimeter_px": total_perimeter_px,
        "MeanDiameter_FWHM_um": mean_fwhm_um,
    }

    if len(df_fwhm):
        df_fwhm["SampleID"] = meta["SampleID"]
        df_fwhm["Condition"] = meta["Condition"]
        df_fwhm["Sex"] = meta["Sex"]
        df_fwhm["File"] = os.path.basename(nc_path)

    return summary, df_fwhm, mask, meta


def process_folder(folder, output_dir, pixel_size_um=1.0, raw_var="tlsc_tmean", mask_var="vessel_mask"):
    os.makedirs(output_dir, exist_ok=True)

    summaries = []
    vessels_all = []
    masks = []
    names = []

    nc_files = sorted([f for f in os.listdir(folder) if f.endswith(".nc")])

    for fname in nc_files:
        full = os.path.join(folder, fname)
        # Pass output_dir to quantify_nc_file
        summary, df_vess, mask, meta = quantify_nc_file(full, pixel_size_um, raw_var, mask_var, output_dir=output_dir)
        summaries.append(summary)
        vessels_all.append(df_vess)
        masks.append(mask)
        names.append(fname)

    df_summary = pd.DataFrame(summaries)
    df_vessels = pd.concat(vessels_all, ignore_index=True)

    df_summary.to_csv(os.path.join(output_dir, "all_masks_summary.csv"), index=False, sep=";")
    df_vessels.to_csv(os.path.join(output_dir, "all_vessels_diameters.csv"), index=False, sep=";")

    return df_summary, df_vessels, masks, names

#plot all masks in a grid
def plot_all_masks_grid(masks, names, output_dir, cols=4):
    n = len(masks)
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5))
    axes = axes.ravel()

    for ax, mask, name in zip(axes, masks, names):
        ax.imshow(mask, cmap="gray")
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_masks.png"), dpi=300)
    plt.close()

#plot boxplot with p-value
def boxplot_with_p(ax, data_control, data_diabetic, ylabel, colors=("skyblue", "salmon")):
    t_stat, p_val = ttest_ind(data_control, data_diabetic, equal_var=False, nan_policy="omit")
    bp = ax.boxplot([data_control, data_diabetic], patch_artist=True, labels=["Control", "Diabetic"])
    bp["boxes"][0].set_facecolor(colors[0])
    bp["boxes"][1].set_facecolor(colors[1])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} (p = {p_val:.4f})")

#plot all metrics
def plot_all_metrics(df_summary, output_dir):
    ctrl = df_summary[df_summary["Condition"] == "control"]
    diab = df_summary[df_summary["Condition"] == "diabetic"]

    metrics = [
        ("FractionVessel_percent", "Vessel Fraction (%)"),
        ("NumVessels", "Number of Vessels"),
        ("TotalPerimeter_px", "Total Perimeter (px)"),
        ("MeanDiameter_FWHM_um", "Mean Diameter (FWHM, µm)")
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for ax, (col, label) in zip(axes, metrics):
        boxplot_with_p(ax, ctrl[col], diab[col], label)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_comparison_all_metrics.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    folder = "/Users/aida/Desktop/APM/time_averaged_tLSC_and_vessel_masks" # folder with nc files (images and masks)
    output_dir = "/Users/aida/Desktop/APM/quant_results_FWHM_only" # folder to save results
    pixel_size_um = 1.0

    RAW_VAR = "tlsc_tmean"
    MASK_VAR = "vessel_mask"

    df_summary, df_vessels, masks, names = process_folder(
        folder, output_dir, pixel_size_um, RAW_VAR, MASK_VAR
    )

    plot_all_masks_grid(masks, names, output_dir)
    plot_all_metrics(df_summary, output_dir)

    print("Analysis complete.")