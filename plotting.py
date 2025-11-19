from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plotResults(figures):
    """
    Generalized function to handle plotting of figures.
    
    Parameters:
        figures (list): A list of dictionaries, where each dictionary contains the following keys:
            - 'data': The data to be plotted (e.g., image, mask, etc.).
            - 'title': Title of the plot.
            - 'cmap': Colormap to be used for the plot (optional, default is None).
            - 'alpha': Transparency level for overlay plots (optional, default is 1.0).
            - 'colorbar': Boolean indicating whether to add a colorbar (optional, default is False).
            - 'overlay': Data to overlay on the main plot (optional, default is None).
    """
    num_plots = len(figures)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]  # Ensure axes is iterable for a single plot

    for ax, fig_data in zip(axes, figures):
        data = fig_data.get('data')
        title = fig_data.get('title', '')
        cmap = fig_data.get('cmap', None)
        alpha = fig_data.get('alpha', 1.0)
        colorbar = fig_data.get('colorbar', False)
        overlay = fig_data.get('overlay', None)

        im = ax.imshow(data, cmap=cmap, alpha=alpha)
        ax.set_title(title)
        ax.axis('off')  # Hide axes for better visualization

        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if overlay is not None:
            ax.imshow(overlay, cmap='gray', alpha=0.4)

    plt.tight_layout()
    plt.show()

    pass

def linearHistogram(img, bins=100, bg_value=1.0, bg_tol=1e-6, 
                             bg_threshold=None, title=None):
    """
    Plot a linear histogram of intensities excluding background.

    Parameters
    ----------
    img : np.ndarray or xarray.DataArray
        Normalized image in [0,1].
    bins : int
        Number of histogram bins.
    bg_value : float
        Background value (use 1.0 for white background).
    bg_tol : float
        Tolerance around bg_value when using isclose.
    bg_threshold : float or None
        If set (e.g., 0.98), exclude pixels >= bg_threshold instead of exact isclose.
    title : str or None
        Plot title.
    """
    arr = img.values if hasattr(img, "values") else img
    arr = np.asarray(arr, dtype=np.float32)

    mask = np.isfinite(arr)
    if bg_threshold is not None:
        mask &= (arr < bg_threshold)        # drop near-white tail
    else:
        mask &= ~np.isclose(arr, bg_value, atol=bg_tol)

    vals = arr[mask]
    if vals.size == 0:
        print("No non-background pixels to plot.")
        return

    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=bins, edgecolor="black")
    plt.xlabel("Intensity")
    plt.ylabel("Count")          # linear scale by default
    plt.xlim(0, 1)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()

    pass


def plotImage(ds, da, plot=False, stats=False):
    """
    Used in early stages of preprocessing to assess the pixel distribution
    """
    # Categorize into zones based on intensity
    bins = [0.09, 0.129, 0.1317, 0.15, 0.28]
    labels = ["Loop (fast flow)", "Smaller vessels", "Static tissue", "Big static"]
    zones = np.digitize(da, bins, right=True)
    
    cmap = ListedColormap(["#2b83ba", "#abdda4", "#ffffbf", "#fdae61"])  # 4 colors

    # Prepare figures for plotting
    figures = [
        {'data': ds["tlsc_tmean"].values, 'title': "TLSC mean image - original", 'cmap': "gray", 'colorbar': True},
        {'data': np.histogram(ds["tlsc_tmean"].values, bins=100)[0], 
         'title': "Histogram of tlsc_tmean intensities", 'cmap': None},
        {'data': zones, 'title': "TLSC categories (flow zones)", 'cmap': cmap, 'colorbar': True}
    ]

    # Use plotResults to display the plots
    if stats:
        imgStat(ds)


    if plot:
        plotResults(figures)

    pass

def imgStat(ds):
    # Compute and print statistics
    vals = ds["tlsc_tmean"].values.flatten()
    print("Mean:", vals.mean())
    print("Median:", np.median(vals))
    print("Std dev:", vals.std())
    print("Min:", vals.min(), "Max:", vals.max())
    pass

def vesselHist(img, bins = 100, vmax = 0.25, title = None, show_stats = False):
    vals = img[np.isfinite(img)]
    vals = vals[(vals >= 0.0) & (vals <= vmax)]    
    vals = (vals - vals.min()) / (vals.max() - vals.min())

    # class masks within the restricted values
    big_mask   = vals <= 0.15
    small_mask = (vals > 0.15) & (vals <= 0.25)

    # 2) plot histogram (linear)
    edges = np.linspace(0.0, vmax, bins+1)
    plt.figure(figsize=(6,4))
    plt.hist(vals, bins=edges, edgecolor="black")
    plt.xlabel("Intensity")
    plt.ylabel("Count")
    plt.xlim(0, vmax)

   # shade class ranges
    plt.axvspan(0.0, 0.15, color="#E63946", alpha=0.15, label="Big vessels [0–0.15]")
    plt.axvspan(0.15, 0.25, color="#4EA8DE", alpha=0.15, label="Small vessels (0.15–0.25]")

    if title:
        plt.title(title)

    if show_stats:
        n_all   = vals.size
        n_big   = int(big_mask.sum())
        n_small = int(small_mask.sum())
        pct_big   = 100.0 * n_big / n_all
        pct_small = 100.0 * n_small / n_all
        txt = f"big: {n_big} ({pct_big:.1f}%)   small: {n_small} ({pct_small:.1f}%)   total in [0–0.25]: {n_all}"
        plt.text(0.5*vmax, 0.95*plt.gca().get_ylim()[1], txt, ha="center", va="top", fontsize=9)

    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.show()

    pass
