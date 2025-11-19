from plotting import plotResults
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma, denoise_tv_chambolle
from scipy.ndimage import spline_filter
import numpy as np

"""
    Methods applied:
        - Bilateral filter
        - TV-Chambolle smoothing
        - B-spline smoothing
        - FFT low-pass filtering    

    Tested: bilateral filter
    To implement: Wavelet, diffusion-based, matched filtering, vesselness-based approaches (from the article)
    TODO: compare all methods and choose the best one
"""

def smoothBSpline(ds_normalized, plot=False):
    """
    Apply B-spline smoothing to the normalized single frame image.
    Parameters:
        ds_normalized (xarray.DataArray): Normalized image data.
    Returns:
        ndarray: B-spline smoothed image.
    """
    arr = ds_normalized.values if hasattr(ds_normalized, 'values') else ds_normalized
    #imgn = ds_normalized.values
    imgn = arr
    img_bspline = spline_filter(imgn, order=3)

    if plot:
        plotCompare(ds_normalized, img_bspline, "B-spline smoothing")

    return img_bspline

def plotCompare(ds_normalized, img_smoothed, method_name):
    """
    Compare original normalized image with smoothed image.
    Parameters:
        ds_normalized (xarray.DataArray): Normalized image data.
        img_smoothed (ndarray): Smoothed image data.
        method_name (str): Name of the smoothing method for title.
    """
    imgn = ds_normalized.values

    figures = [
        {'data': imgn, 'title': "Original normalized", 'cmap': "gray", 'colorbar': True},
        {'data': img_smoothed, 'title': f"{method_name} smoothed", 'cmap': "gray", 'colorbar': True}
    ]

    plotResults(figures)