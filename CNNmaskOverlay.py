import __main__
import matplotlib
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, RectangleSelector, Button
from matplotlib.patches import Rectangle
#matplotlib.use("TkAgg")   # use Tk backend to avoid Qt error
import xarray as xr
import numpy as np
from scipy import signal
from scipy.ndimage import binary_dilation, spline_filter, label
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma
from skimage.filters import sato, frangi, meijering, threshold_otsu
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import remove_small_objects, remove_small_holes, binary_erosion, binary_dilation
from scipy.signal import welch, butter, filtfilt, hilbert, savgol_filter
import seaborn as sns
import pandas as pd
import os


# --- Path to your file ---
path = 'D:/APM_Project/'
filename = path + '20241011b_control_f_vehicle_adult.nc' 
mask = path + 'meanImagesMasks/subject01_mask20241011b_control_f_vehicle_adult_mask.nc'
#filename = path + 'timeAveragedtLSC/20241011b_control_f_vehicle_adult.nc'
#filename = path + 'APM_KidneyProject-Natalia_new/APM_KidneyProject-Natalia_new/20241011b_control_f_vehicle_adult.nc'
#filename = path + '20241028a_diabet_m_vehicle_adult.nc'
#mask = path + 'meanImagesMasks/subject07_mask_20241028a_diabetc_m_vehicle_adult.nc'
save_dir = path + "saved_signals/"
os.makedirs(save_dir, exist_ok=True)

sns.set_theme(style="darkgrid", palette="deep")
plt.rcParams.update({
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.2,
    "legend.fontsize": 12,
    "figure.figsize": (9, 4.5),
    "grid.alpha": 1.0,
    "grid.color": "#333333",
    "grid.linewidth": 0.8,
})

REPORT_DIR = os.path.join(path, "ReportPlots")
os.makedirs(REPORT_DIR, exist_ok=True)

def detect_dataset_label(filepath):
    """Return 'control', 'diabetic', or 'unknown' based on filename."""
    fp = filepath.lower()
    if "control" in fp:
        return "control"
    if "diabet" in fp:
        return "diabetic"
    return "unknown"

# dataset_label already defined earlier, but ensure global consistency:
dataset_label = detect_dataset_label(filename)

def save_figure(basename, fig=None):
    """
    Save a matplotlib figure with auto-numbering and dataset label.
    
    Parameters
    ----------
    basename : str
        Base name, e.g. "RawSignal", "PSD", "FilteredSignal", "SelectedVesselOverlay"
    fig : matplotlib.figure.Figure or None
        If None, uses the current active figure (plt.gcf()).
    """

    if fig is None:
        fig = plt.gcf()

    # Count existing files for auto-numbering
    existing = [
        f for f in os.listdir(REPORT_DIR)
        if f.endswith(".png") and basename in f
    ]
    num = len(existing) + 1

    # Construct final filename
    filename_out = f"{num:02d}_{dataset_label}_{basename}.png"
    save_path = os.path.join(REPORT_DIR, filename_out)

    # Save
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✔ Saved figure: {save_path}")

def openDatasetWithMask(filename, mask, x_slice, y_slice, t_slice, x_shift=0, y_shift=0):
    
    ds = xr.open_dataset(filename, decode_timedelta=True)
    mask_ds = xr.open_dataset(mask)
    
    # Check variable name
    data_var = list(ds.data_vars.keys())[0]
    mask_var = list(mask_ds.data_vars.keys())[0]
    
    # 1) Normalize/rename dataset dims so we have canonical names 't','y','x'
    #    Map any dim whose name starts with 'y' -> 'y', 'x' -> 'x', 't' -> 't'
    def canonical_map(dims):
        mapping = {}
        for d in dims:
            d0 = d.lower()
            if d0.startswith('t'):
                mapping[d] = 't'
            elif d0.startswith('y') or d0.startswith('row'):
                mapping[d] = 'y'
            elif d0.startswith('x') or d0.startswith('col'):
                mapping[d] = 'x'
        return mapping

    ds = ds.rename(canonical_map(ds[data_var].dims))
    
    da = ds[data_var].transpose('t', 'y', 'x')
    
    mask_da = mask_ds[mask_var]
    mask_da = mask_da.rename(canonical_map(mask_da.dims))
        
    # Ensure dimensions match in name and order
    # If mask has (y, x) but dataset has (x, y), reorder:
    if list(mask_da.dims) != ['y', 'x']:
        mask_da = mask_da.transpose('y', 'x')
        
    mask_array = mask_da.values.astype(float)
    
    # Align mask to dataset dimensions
    ny, nx = ds[data_var].isel(t=0).shape
    my, mx = mask_array.shape
    
    aligned_mask = np.zeros((ny, nx), dtype=float)
    
    # Place mask aligned to top-right corner
    y_start = max(0, (ny - my) // 2 + y_shift)
    x_start = max(0, (nx - mx) // 2 + x_shift)
    
    # Compute end positions within bounds
    y_end = min(y_start + my, ny)
    x_end = min(x_start + mx, nx)
    
    # Compute actual mask region to copy (avoid mismatch)
    mask_y_end = min(my, y_end - y_start)
    mask_x_end = min(mx, x_end - x_start)
    
    aligned_mask[y_start:y_start + mask_y_end, x_start:x_start + mask_x_end] = \
        mask_array[:mask_y_end, :mask_x_end]
        
    # Display full image
    first_frame_full = ds[data_var].isel(t=0).values.astype(np.float64)
    plt.figure(figsize=(8, 8))
    plt.imshow(first_frame_full, cmap='gray', origin='lower')
    plt.imshow(aligned_mask, cmap='Reds', alpha=0.4, origin='lower')
    plt.title("First frame of full dataset with mask overlay")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.axis('on')
    plt.show()
    
    # Crop small region
    ds_small = ds.isel(y=y_slice, x=x_slice).sel(t=t_slice).load() 
    mask_small = aligned_mask[y_slice, x_slice].astype(bool)
    
    mean = ds_small['tlsc'].mean('t').values.astype(np.float64)
    #ds_small['tlsc'].mean('t').plot.imshow() #  we see darker regions, representing surface vessels, and other details
    plt.figure(figsize=(6, 6))
    plt.imshow(mean, cmap='gray', origin='lower')
    plt.imshow(mask_small, cmap='Reds', alpha=0.4, origin='lower')
    plt.title("Mean image with segmentation mask overlay")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.axis('on')
    plt.show()
    
    return ds_small, mask_small, mean

def selectSingleVessel(mask_small, image=None):
    """
    Select a single vessel from a binary mask by clicking on it.
    
    Parameters
    ----------
    mask : 2D np.array (bool or 0/1)
        Binary mask of vessels (white = vessels)

    Returns
    -------
    selected_vessel : 2D bool np.array
        Mask of the selected vessel
    """
    #matplotlib.use("TkAgg")  # switch to Tk for interactive click

    labeled_mask, num_features = label(mask_small)  # label connected components
    print(f"Found {num_features} vessels")
    
    selected_vessel = np.zeros_like(mask_small, dtype=bool)
    
    root = tk.Tk()
    root.title("Select a vessel")

    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    im_display = ax.imshow(mask_small, cmap='gray', origin='lower')
    #ax.set_title("Click on a vessel to select it")
    confirmed = [False]
    
    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        label_id = labeled_mask[y, x]
        if label_id == 0:
            print("Clicked background. Click on a vessel.")
            return
        selected_vessel[:,:] = labeled_mask == label_id
        im_display.set_data(selected_vessel)
        im_display.set_cmap('grey')
        fig.canvas.draw()
#        nonlocal selected_vessel
        print(f"Vessel {label_id} selected, press OK to confirm.")
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    def on_ok():
        confirmed[0] = True
        root.destroy()
    
    btn = tk.Button(root, text="OK", command=on_ok)
    btn.pack()
    
    root.mainloop()
#    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
#    ax_ok = plt.axes([0.8, 0.01, 0.1, 0.05])
#    btn = Button(ax_ok, 'OK')
#    btn.on_clicked(on_ok)
    
#    plt.show()
    
    if confirmed[0]:
        # Restore inline backend
        plt.close(fig)
        matplotlib.use("module://ipykernel.pylab.backend_inline")      
        
        if image is not None:
            plt.figure(figsize=(7, 6))
            plt.imshow(image, cmap='plasma', origin='lower')
            plt.imshow(selected_vessel, cmap='gray', alpha=0.5, origin='lower')
            plt.title("Selected Vessel Overlay")
            plt.axis('off')
            save_figure("SelectedVessel")
            plt.show()

    else:
        print("Selection canceled or window closed without pressing OK")
        matplotlib.use("module://ipykernel.pylab.backend_inline")
        return None

    return selected_vessel

def vesselMeanOverTime(ds_small, selected_vessel, var="tlsc"):
    """
    Compute and plot the mean intensity over time within the selected vessel.

    Parameters
    ----------
    ds_small : xarray.Dataset
        Dataset containing the time series (e.g., variable 'tlsc').
    selected_vessel : 2D bool np.ndarray
        Mask of the selected vessel (True = inside vessel).
    var : str, optional
        Variable name in the dataset (default: 'tlsc').

    Returns
    -------
    mean_over_time : np.ndarray
        Mean intensity values over time inside the selected vessel.
    time_vals : np.ndarray
        Time coordinate values.
    """
    da = ds_small[var]
    
    if 't' in da.coords:
        da = da.assign_coords(t = da.t.astype(float) * 1e-9)
        
    time_dim= 't'
 
     # Check spatial dimensions (assume y, x order) 
    spatial_dims = [d for d in da.dims if d != time_dim]
    y_dim, x_dim = spatial_dims

    # --- Broadcast mask to dataset shape and apply it ---
    mask_da = xr.DataArray(selected_vessel, dims=(y_dim, x_dim))
    masked_da = da.where(mask_da)

    # Compute mean over spatial dimensions (only vessel pixels)
    mean_over_time = masked_da.mean(dim=(y_dim, x_dim), skipna=True).compute()

    # Time values
    time_vals = da[time_dim].values

    # Plot time series
    plt.figure(figsize=(9, 4))
    sns.lineplot(x=time_vals, y=mean_over_time.values, color='royalblue')
    plt.xlabel("Time (s)")
    plt.ylabel("Mean intensity in selected vessel")
    plt.title("Temporal variation of vessel intensity")
    plt.tight_layout()
    save_figure("RawSignal")
    plt.show()

    return mean_over_time.values, time_vals

def selectMultipleVessels(mask_small, image=None):
    """
    Allows selecting multiple vessels by clicking on them.
    Selected vessels are added cumulatively.
    Returns a list of individual vessel masks.
    """
    labeled_mask, num_features = label(mask_small)
    print(f"Found {num_features} vessels")

    selected_mask = np.zeros_like(mask_small, dtype=bool)
    selected_labels = set()

    root = tk.Tk()
    root.title("Select Multiple Vessels (Click to Select)")

    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    im_display = ax.imshow(mask_small, cmap='gray', origin='lower')

    confirmed = [False]

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        x, y = int(event.xdata), int(event.ydata)
        label_id = labeled_mask[y, x]

        if label_id == 0:
            print("Clicked background. Click on a vessel.")
            return

        # Add vessel to selection
        if label_id not in selected_labels:
            selected_labels.add(label_id)
            selected_mask[labeled_mask == label_id] = True
            print(f"Added vessel {label_id} to selection.")
        else:
            # Allow deselecting
            selected_labels.remove(label_id)
            selected_mask[labeled_mask == label_id] = False
            print(f"Removed vessel {label_id} from selection.")

        # Update display
        overlay = mask_small.astype(float)
        overlay[selected_mask] = 1.5  # highlight selected
        im_display.set_data(overlay)
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)

    def on_ok():
        confirmed[0] = True
        root.destroy()

    btn = tk.Button(root, text="OK", command=on_ok)
    btn.pack()

    root.mainloop()
    plt.close(fig)

    if not confirmed[0]:
        print("Selection cancelled.")
        return None

    # Return list of individual vessel masks
    vessel_masks = [(labeled_mask == lab) for lab in selected_labels]
    return vessel_masks

def analyze_multiple_vessels(ds_small, vessel_masks, time_vals, var="tlsc"):
    """
    For each mask:
    - compute mean signal over time
    - compute PSD
    - extract dominant frequency & amplitude
    - plot power spectrum
    """

    def compute_psd(sig, fs):
        freqs, psd = welch(sig, fs=fs, nperseg=256)
        return freqs, psd

    fs = 1 / np.mean(np.diff(time_vals))
    results = []

    for i, mask in enumerate(vessel_masks, start=1):
        print(f"\n--- Vessel {i} ---")

        # Extract signal
        da = ds_small[var]
        spatial_dims = [d for d in da.dims if d != "t"]
        y_dim, x_dim = spatial_dims

        mask_da = xr.DataArray(mask, dims=(y_dim, x_dim))
        masked_da = da.where(mask_da)
        mean_ts = masked_da.mean(dim=(y_dim, x_dim)).values

        # Compute PSD
        freqs, psd = compute_psd(mean_ts, fs)

        # Dominant frequency
        idx = np.argmax(psd)
        dominant_freq = freqs[idx]
        amplitude = np.sqrt(psd[idx])

        print(f"Dominant frequency: {dominant_freq:.4f} Hz")
        print(f"Amplitude: {amplitude:.4f}")

        # Plot PSD
        plt.figure(figsize=(6,4))
        plt.plot(freqs, psd)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title(f"Power Spectrum – Vessel {i}")
        plt.grid(True)
        plt.show()

        results.append({
            "vessel_index": i,
            "dominant_frequency": dominant_freq,
            "amplitude": amplitude
        })

    return results

# ANALYSIS OF THE SIGNAL

def analyze_frequency(ds_small, vessel_mask, time_vals, var="tlsc"):
    """
    Analyze one vessel:
    - compute mean signal over time
    - compute PSD (Welch)
    - extract dominant frequency & amplitude
    - plot the power spectrum
    """

    # Sampling frequency
    fs = 1 / np.mean(np.diff(time_vals))

    # Extract dataset variable
    da = ds_small[var]

    # Detect spatial dimensions
    spatial_dims = [d for d in da.dims if d != "t"]
    y_dim, x_dim = spatial_dims

    # Apply mask
    mask_da = xr.DataArray(vessel_mask, dims=(y_dim, x_dim))
    masked_da = da.where(mask_da)

    # Compute mean signal over time
    mean_ts = masked_da.mean(dim=(y_dim, x_dim)).values

    # Compute PSD
    freqs, psd = welch(mean_ts, fs=fs, nperseg=256)

    # Dominant frequency
    idx = np.argmax(psd)
    dominant_freq = freqs[idx]
    amplitude = np.sqrt(psd[idx])

    # Print results
    print("\n=== Single Vessel Frequency Analysis ===")
    print(f"Dominant frequency: {dominant_freq:.4f} Hz")
    print(f"Amplitude:          {amplitude:.4f}")

    # Plot PSD
    plt.figure(figsize=(7,4))
    sns.lineplot(x=freqs, y=psd, color='darkblue')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Power Spectrum – Selected Vessel")
    plt.tight_layout()
    save_figure("PowerSpectrum")
    plt.show()

    return {
        "dominant_frequency": dominant_freq,
        "amplitude": amplitude,
        "freqs": freqs,
        "psd": psd,
        "mean_signal": mean_ts
    }


# FILTERING

def detrend_signal(signal, time):
    """Remove linear trend."""
    p = np.polyfit(time, signal, 1)
    trend = np.polyval(p, time)
    detrended = signal - trend
    return detrended, trend

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Butterworth band-pass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_tgf_filtered(signal, time, band=(0.02, 0.05)):
    """Detrend + TGF band-pass filter."""
    fs = 1 / np.mean(np.diff(time))
    detrended, trend = detrend_signal(signal, time)
    filtered = butter_bandpass_filter(detrended, band[0], band[1], fs)
    return detrended, filtered

def plot_tgf_filtered(signal, filtered, time):
    plt.figure(figsize=(9,4))
    sns.lineplot(x=time, y=signal, label='Original signal', color='#4A90E2', alpha=0.5)
    sns.lineplot(x=time, y=filtered, label='Filtered signal (0.02–0.05 Hz)', color='#E67E22')
    plt.xlabel("Time (s)")
    plt.ylabel("Mean intensity (a.u.)")
    plt.title("Temporal intensity variation in vessel")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9,4))
    sns.lineplot(x=time, y=filtered, color='#E67E22')
    plt.xlabel("Time (s)")
    plt.ylabel("Signal intensity")
    plt.title("TGF-filtered signal (0.02–0.05 Hz)")
    plt.tight_layout()
    save_figure("FilteredSignal")
    plt.show()
    
def save_signal(time_vals, mean_over_time, name=None):
    """
    Save filtered vessel signal to disk as a compressed .npz file.
    Automatically adds dataset label (control/diabetic) based on filename.
    """

    # --- Auto-generate filename if not provided ---
    if name is None:
        existing = [
            f for f in os.listdir(save_dir)
            if f.startswith(f"signal_{dataset_label}_")
        ]
        num = len(existing) + 1
        name = f"signal_{dataset_label}_{num:02d}.npz"

    # --- Save to disk ---
    np.savez(os.path.join(save_dir, name), time=time_vals, signal=mean_over_time)
    print(f"✔ Saved filtered signal as: {name}  (dataset: {dataset_label})")

if __name__ == "__main__":
    ds_small, mask_small, mean = openDatasetWithMask(filename, mask, 
                        x_slice=slice(400, 1300), y_slice=slice(400, 1300),
                        t_slice=slice('0s', '800s'), x_shift=-5, y_shift=1)
    
    selected_vessel = selectSingleVessel(mask_small, mean)
    mean_vessel_ts, time_vals = vesselMeanOverTime(ds_small, selected_vessel, var="tlsc")
    results = analyze_frequency(ds_small, selected_vessel, time_vals, var="tlsc")
    #smoothed_signal = filterSignal(mean_vessel_ts, time_vals, method='savgol', window_length=15, polyorder=3)
    detrended, filtered = extract_tgf_filtered(mean_vessel_ts, time_vals)
    save_signal(time_vals, mean_vessel_ts)
    plot_tgf_filtered(mean_vessel_ts, filtered, time_vals)
    
'''
    vessel_masks = selectMultipleVessels(mask_small, mean)

    # Compute the time-series of ds_small once
    _, time_vals = vesselMeanOverTime(ds_small, np.ones_like(mask_small), var="tlsc")

    results = analyze_multiple_vessels(ds_small, vessel_masks, time_vals)

'''