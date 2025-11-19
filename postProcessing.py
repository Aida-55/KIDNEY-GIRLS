from matplotlib.colors import ListedColormap
from plotting import plotResults

from skimage.filters import sato, frangi, meijering
from skimage.morphology import remove_small_objects, remove_small_holes, binary_dilation, binary_erosion
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes

def binarize(img, thr = 0.1, plot = True):
    imgBin = img <= thr

    figures = [
        {'data': imgBin, 'title': 'binarized image', 'cmap': 'gray'},
        {'data': img, 'title': 'Segmented vessels overlay', 'cmap': 'gray', 'overlay': imgBin}
    ]

    if plot:
        plotResults(figures)

    return imgBin
    
def morphologicalOps(mask, original_img, px_um = 1.0, min_obj_area_um2 = 180.0, plot = False):
    """
    Post-processing: Apply morphological operations to clean up the vessel mask.
    Parameters:
        mask (ndarray): Binary mask of vessels. 
        original_img (ndarray): Original image for overlay visualization.
        px_um (float): Pixel size in micrometers.
        min_obj_area_um2 (float): Minimum object area in square micrometers to keep.
    Returns:
        ndarray: Cleaned binary mask after morphological operations.
        """
    # 1 px area in µm²
    min_obj_area_px = int(np.ceil(min_obj_area_um2 / (px_um**2)))
    mask_bool = mask.astype(bool)

    # Morphological operations to clean up the mask
    mask_clean = binary_erosion(mask)
    mask_clean = remove_small_objects(mask_bool, min_size=min_obj_area_px,)  # try 100–300 px
    mask_clean = binary_dilation(mask_clean, footprint=np.ones((1,1)))
    mask_clean = remove_small_holes(mask_clean, area_threshold=min_obj_area_px)
    mask_clean = binary_fill_holes(mask_clean)
    #mask_clean = binary_erosion(mask_clean, footprint=np.ones((1,2)))
    mask_clean = binarize(mask_clean)
    # Use plotResults to visualize results
    figures1 = [
        {'data': mask, 'title': 'Before cleaning', 'cmap': 'gray'},
        {'data': mask_clean, 'title': 'After removing small objects', 'cmap': 'gray'}
    ]

    figures2 = [
        {'data': original_img, 'title': 'Original image', 'cmap': 'gray'},
        {'data': mask_clean, 'title': 'Mask of enhanced vessels', 'cmap': 'gray'},
        {'data': original_img, 'title': 'Segmented vessels overlay', 'cmap': 'gray', 'overlay': mask_clean}
    ]

    if plot:
        plotResults(figures1)
        plotResults(figures2)   

    return mask_clean
