from plotting import plotResults
import numpy as np


def normalizeWithThreshold(da, thr, plot = False):
    """
    Preprocess a single frame dataset.
    Goal is to cut off background and artifact values, leaving the image with enhanced contrast for vessels.
    Parameters:
        da (xarray.Dataset): Input dataset containing the image data. (as type float32)

    Returns:        
        xarray.Dataset: The original dataset.
        ndarray: The normalized image data.
    """
    # Cut values in the image to [da.min, thr] range 
    da_norm = da.where(da <= thr, thr)

    # Normalize to [0, 1] range
    da_norm = (da_norm - da_norm.min()) / (da_norm.max() - da_norm.min())

    figures = [
        {'data': da_norm, 'title': "TLSC mean image - normalized", 'cmap': "gray", 'vmin': 0, 'vmax': 1},
        {'data': da, 'title': "Original TLSC image", 'cmap': "gray", 'colorbar': True}
    ]
    
    if plot:
        plotResults(figures)

    return da_norm

def normalizeWithPercentile(img_or_da, p_lo=1, p_hi=99, plot=False):
    """
    Normalize a single frame dataset using 1st and 99th percentiles.

    Parameters: 
        img_or_da (NumPy array or xarray.DataArray): DataArray containing the image data. (float32), 
                         if not the type - converts to dataArray
        p_lo=1, p_hi=99 — the lower and upper percentiles used as black/white points.
    Returns:
        ndarray: The normalized image data.
    
    """
    # Check for data type
    arr = img_or_da.values if hasattr(img_or_da, "values") else img_or_da
    arr = np.asarray(arr, dtype=np.float32)

    # 1) keep only finite values for stats (no NaN, inf, -inf)
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("percent_norm_safe: no finite values in input.")
    vals = arr[finite]

    # lo - black point, hi - white point
    lo = np.percentile(vals, p_lo)
    hi = np.percentile(vals, p_hi)

    # 2) if p_lo/p_hi collapsed, widen or fall back
    if hi <= lo:
        # fall back: use min/max; if still degenerate, return zeros
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)

    # 3) clip and scale
    imgc = np.clip(arr, lo, hi)
    imgn = (imgc - lo) / (hi - lo)

    # 4) sanitize any residual non-finites
    imgn = np.nan_to_num(imgn, nan=0.0, posinf=1.0, neginf=0.0)
    imgNorm = imgn

    # Use plotResults to display before and after normalization
    figures = [
        {'data': img_or_da, 'title': "Original TLSC image", 'cmap': "gray", 'colorbar': True},
        {'data': imgNorm, 'title': "Normalized (0–1)", 'cmap': "gray", 'colorbar': True}
    ]

    if plot:
        plotResults(figures)

    return imgNorm


