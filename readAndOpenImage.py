import os
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import xarray as xr
import numpy as np
from plotting import plotResults

"""
Generalized function to handle opening and displaying a single frame from a dataset.
Methods:
- openFile: Opens a dataset and extracts relevant data.
- openFolder: Displays all .nc images in a specified folder.
"""

def openFile(filename):
    """
    Open and display a single frame from the dataset. 
    Does simple analysis and plotting.

    Parameters:
        filename (str): Path to the .nc dataset file.
    Returns:
        xarray.Dataset: The opened dataset.
        """
    # Read and open dataset
    ds = xr.open_dataset(filename)
    print(ds)
    px_um = float(ds.attrs.get("pixel_width_um", 1.0))
    
    # Extract data array and convert to float32
    da = ds["tlsc_tmean"].astype("float32")

    return ds, da, px_um

def openFolder(path, plot = False):
    """
    Display all .nc images in the specified folder in a single window with 4 columns and 2 rows,
    including a color bar next to each plot.

    Parameters:
        path (str): The directory path containing .nc image files.

    Returns:
        list: List of .nc image filenames found in the directory.
    """
    
    # View all .nc images in the specified folder - there are 8 images
    image_files = [f for f in os.listdir(path) if f.endswith('.nc')]

    # Read and display in one window all images 4 columns, 2 rows
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 6), gridspec_kw={'wspace': 0.6, 'hspace': 0.4})
    axes = axes.flatten()
    for ax, img_file in zip(axes, image_files):
        ds = xr.open_dataset(os.path.join(path, img_file))
        img = ds["tlsc_tmean"]
        im = ax.imshow(img, cmap="gray", origin="lower", aspect='equal')
        ax.set_title(f"{img_file}", fontsize=8)
        ax.axis('off')  # Hide axes for better visualization

        # Add a color bar next to each plot
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    #plt.tight_layout()
    if plot == True:
        plt.show()

    return image_files