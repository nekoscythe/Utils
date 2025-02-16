import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .helper_plotting_utils import get_optimal_subplot_dims, create_subplots, get_colormap_and_limits, make_cbar
from .data_manipulation_utils import interpolate_to_rho, slice_data
from .depth_utils import get_zlevs, get_depth_index


def compare_datasets(datasets, variable, time_idx, eta=None, xi=None, suptitle=None, z_levs=None, titles=None, max_depth=500, gen_levels=False, vmin=None, vmax=None, s_rho=-1, max_columns=None):
    """
    Compares and plots a variable from multiple datasets side-by-side.

    This function generates a grid of subplots, with each column representing a dataset and each row
    (optionally, if time_idx is a list) representing a time index. It can plot either surface maps
    or depth profiles depending on whether 'eta' or 'xi' is specified.

    Parameters:
        datasets (list of xarray.Dataset): A list of datasets to compare.
        variable (str): The name of the variable to plot.
        time_idx (int or list of int): Time index or list of time indices to plot.
        eta (int, optional): Index along latitudinal dimension (for zonal depth profile). Defaults to None (surface plot).
        xi (int, optional): Index along longitudinal dimension (for meridional depth profile). Defaults to None (surface plot).
        suptitle (str, optional): Overall suptitle for the figure. Defaults to None.
        z_levs (np.ndarray, optional): Depth levels array (if available). Defaults to None.
        titles (list of str, optional): Titles for each column (dataset). Defaults to None.
        max_depth (float, optional): Maximum depth to plot for depth profiles. Defaults to 500m.
        gen_levels (bool, optional): Whether to automatically generate contour levels for depth plots. Defaults to False.
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        s_rho (int, optional): s_rho level for surface plots. Defaults to -1 (surface).
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to None (auto-determined).

    Returns:
        list of matplotlib.axes.Axes: A list of axes objects for the generated subplots.
    """

    #if time_idx is an integer, convert to list
    if isinstance(time_idx, int):
        time_idx = [time_idx]

    n_datasets = len(datasets)
    n_times = len(time_idx)

    fig, ax = create_subplots(n_datasets * n_times , max_columns=n_datasets if max_columns is None else max_columns)
    # plt.subplots_adjust(right=0.8)  # Make room on the right for the colorbar

    s_rho = None if eta is not None or xi is not None else s_rho
    if vmin is None or vmax is None:
        minmax = get_minmax_datasets(datasets, variable, time=time_idx, eta=eta, xi=xi, s_rho=s_rho, max_depth=max_depth, z_levs=z_levs)
    minmax = [vmin if vmin is not None else minmax[0], vmax if vmax is not None else minmax[1]]

    ims = [] # list to store im objects
    for j, time in enumerate(time_idx):
        for i, dataset in enumerate(datasets):
            idx = j * n_datasets + i
            if eta is not None or xi is not None: #plot depth
                im = plot_depth(dataset, variable, time, ax=ax[idx], eta=eta, xi=xi, z_levs=z_levs, vmin=minmax[0], vmax=minmax[1], max_depth=max_depth, cbar=False, gen_levels=gen_levels)
            else: #plot surface
                _ , im = plot_surface(dataset, variable, time, ax=ax[idx], vmin=minmax[0], vmax=minmax[1], cbar=False, s_rho=s_rho)
            ims.append(im) # append im to list
            ax[idx].set_aspect('auto')
            ax[idx].label_outer()

    #add a space between suptitle and the plots
        #use titles on the top row only
    if titles is not None:
        for i, title in enumerate(titles):
            ax[i].set_title(title, fontsize=8)
    plt.suptitle(suptitle, fontsize=16)
    plt.tight_layout()

    data_var = datasets[-1][variable]
    plt.subplots_adjust(right=0.85, top= 0.9)  # Make room on the right for the colorbar and top for the suptitle
    # Create a new axis for the colorbar spanning the entire figure
    cbar_ax = fig.add_axes([0.9, 0.05, 0.03, 0.86])  # [left, bottom, width, height]
    # Add the colorbar to the new axis
    cbar = fig.colorbar(ims[-1], cax=cbar_ax) # use the last im for the colorbar
    cbar.set_label(f"{data_var.attrs['units']}")

    return ax


def plot_surface(dataset, variable, time_idx, ax=None, title=None, vmin=None, vmax=None, cbar=True, s_rho=-1):
    """
    Plots a surface map of a given variable at a specific time index.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        variable (str): The name of the variable to plot.
        time_idx (int): The time index to plot.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None (auto-generated).
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        cbar (bool, optional): Whether to display a colorbar. Defaults to True.
        s_rho (int, optional): s_rho level to plot (for 3D variables). Defaults to -1 (surface).

    Returns:
        tuple: A tuple containing:
            - ax (matplotlib.axes.Axes): The axes object on which the plot was created.
            - im (matplotlib.collections.QuadMesh): The image object created by pcolormesh.
    """

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = ax.get_figure()


    var_data = dataset[variable][time_idx, s_rho] if len(dataset[variable].shape) == 4 else dataset[variable][time_idx] # 4D vs 3D data
    data = interpolate_to_rho(var_data)   #Interpolate data to rho grid

    # Determine color scaling
    cmap, vmin, vmax = get_colormap_and_limits(data, vmin, vmax)
    if "lon_km" in dataset and "lat_km" in dataset:
        X, Y = dataset.lon_km.values, dataset.lat_km.values
    else:
        X, Y = dataset.lon_rho.values, dataset.lat_rho.values
    time_in_months = dataset.time.values[time_idx] / (3600 * 24 * 30)  # Convert seconds to months
    im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')

    ax.set_title(title, wrap=True, fontsize = 8)
    ax.set_xlabel('Longitude (Km)')
    ax.set_ylabel('Latitude (Km)')
    ax.set_aspect('equal')

    if title is None:
        title = f"{var_data.attrs['long_name']} at {time_in_months:.1f} months"
    ax.set_title(title, wrap=True, fontsize = 8)

    plt.tight_layout()

    if cbar:
        make_cbar(var_data, ax, im)


    return ax, im


def plot_surfaces(dataset, variable, time_idx, axs=None, shading="gouraud", vmin=None, vmax=None, suptitle=None):
    """
    Plots surface maps of a given variable at multiple time indices.

    Generates a grid of subplots, each showing the surface map at a different time index.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        variable (str): The name of the variable to plot.
        time_idx (int or list of int): Time index or list of time indices to plot.
        axs (list of matplotlib.axes.Axes, optional): List of axes objects to plot on. If None, new axes are created. Defaults to None.
        shading (str, optional): Shading mode for pcolormesh. Defaults to "gouraud".
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        suptitle (str, optional): Overall suptitle for the figure. Defaults to None (auto-generated).

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object.
            - axs (list of matplotlib.axes.Axes): A list of axes objects for the generated subplots.
    """

    # Ensure time_idx is a list
    time_idx = [time_idx] if isinstance(time_idx, int) else time_idx


    # Extract data for each time index
    var_data = dataset[variable]
    data = var_data[time_idx, -1].values

    cmap, vmin, vmax = get_colormap_and_limits(data, vmin, vmax)

    X, Y = dataset.lon_km.values, dataset.lat_km.values

    times_in_months = dataset.time.values[time_idx] / (3600 * 24 * 30)  # Convert seconds to months

    # Set up figure and axes
    if axs is None:
        fig, axs = create_subplots(len(time_idx))
    else:
        axs = axs.flatten()

    im = None # Initialize im here
    # Plot each time index
    for i in range(len(time_idx)):
        _, im = plot_surface(dataset, variable, time_idx[i], ax=axs[i], vmin=vmin, vmax=vmax, cbar=False) # capture im here
        axs[i].label_outer()

    long_name = var_data.attrs.get('long_name', '')

    # Add title
    s_title = f"{long_name} at the surface" if suptitle is None else suptitle
    plt.suptitle(s_title, fontsize=14, x=0.5, y=1)
    plt.tight_layout()

    plt.subplots_adjust(right=0.85, top= 0.9)  # Make room on the right for the colorbar and top for the suptitle
    # Create a new axis for the colorbar spanning the entire figure
    cbar_ax = fig.add_axes([0.87, 0.02, 0.03, 0.9])  # [left, bottom, width, height]
    # Add the colorbar to the new axis
    cbar = fig.colorbar(im, cax=cbar_ax) # use captured im here
    cbar.set_label(f"{var_data.attrs['units']}")

    return fig, axs


def plot_depth_simple(data, z_levels, max_depth=500):
    """
    Plots a simple depth profile (X-Z plot) of the given data.

    This function is a basic depth plotting utility that assumes the input 'data' is already
    a 2D array representing data at different depths, and 'z_levels' are the corresponding depth values.

    Parameters:
        data (np.ndarray): 2D numpy array of data at different depths (depth dimension should be the first dimension).
        z_levels (np.ndarray): 1D array of depth levels corresponding to the rows of 'data' (negative values, increasing downwards).
        max_depth (float, optional): Maximum depth to plot. Defaults to 500m.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object.
            - ax (matplotlib.axes.Axes): The axes object on which the plot was created.
    """
    fig, ax = plt.subplots()
    im = ax.pcolormesh(data, cmap="viridis")
    # ax.set_aspect("auto")
    ax.set_xlabel("X")
    ax.set_ylabel("Depth")
    ax.set_ylim([0, max_depth])
    ax.set_yticks(range(0, max_depth + 1, 50))
    ax.set_yticklabels([f"{z:.0f}" for z in np.interp(ax.get_yticks(), range(0, max_depth + 1, 50), z_levels)])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Data")
    return fig, ax


def plot_depth(dataset, variable, time=0, eta=None, xi=None, max_depth=500, ax=None, title=None, vmin=None, vmax=None, z_levs=None, levels=None, gen_levels=False, cbar=True, shading="gouraud"):
    """
    Plots a depth profile (X-Z or Y-Z plot) of a given variable from the dataset.

    This function extracts a depth slice using 'slice_data' and plots it as a pcolormesh.
    It can create either a zonal (longitude-depth) or meridional (latitude-depth) profile
    depending on whether 'eta' or 'xi' is specified.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        variable (str): The name of the variable to plot.
        time (int, optional): The time index to plot. Defaults to 0.
        eta (int, optional): Index along latitudinal dimension (for zonal depth profile). Defaults to None (meridional profile).
        xi (int, optional): Index along longitudinal dimension (for meridional depth profile). Defaults to None (zonal profile).
        max_depth (float, optional): Maximum depth to plot. Defaults to 500m.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None (auto-generated).
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        z_levs (np.ndarray, optional): Depth levels array (if available). Defaults to None (auto-determined).
        levels (list of float, optional): Contour levels to overlay. Defaults to None (no contours).
        gen_levels (bool, optional): Whether to automatically generate contour levels. Defaults to False.
        cbar (bool, optional): Whether to display a colorbar. Defaults to True.
        shading (str, optional): Shading mode for pcolormesh. Defaults to "gouraud".

    Returns:
        matplotlib.collections.QuadMesh: The image object created by pcolormesh.
    """
    # Create figure and axis if not passed
    fig, ax = plt.subplots() if ax is None else (ax.get_figure(), ax)


    # Data preparation
    data, X, xlabel = slice_data(dataset, variable, time, eta, xi) # Extract data slice
    Y, z_var = get_zlevs(dataset) if z_levs is None else (z_levs, None) # Get z-levels
    if z_var != "s_rho" and z_var != "s_w":
        z_idx = get_depth_index(Y, max_depth)
        data = data[z_idx:]
        Y = Y[z_idx:]

    # Set colormap and limits
    if vmin is None and vmax is None:
        cmap, vmin, vmax = get_colormap_and_limits(data, vmin, vmax)
    else:
        cmap = "turbo" if (vmin >= 0) else "seismic"

    # Plot the data
    im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)

    # Contour levels setup
    contour_levels = np.round(np.linspace(data.min(), data.max(), 10), decimals=1) if gen_levels else (levels if levels is not None else [])
    #remove duplicate levels
    contour_levels = list(dict.fromkeys(contour_levels))
    cs = ax.contour(X, Y, data, levels=contour_levels, colors="black", linestyles="solid")
    ax.clabel(cs, inline=True, fontsize=8)

    # Labeling
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (m)")
    if z_var != "s_rho" and z_var != "s_w":
        ax.set_ylim([-max_depth, 0])

    # Title with time in months
    time_in_months = dataset.time.values[time] / (3600 * 24 * 30)
    ax.set_title(title or f"{dataset[variable].attrs['long_name']} at {time_in_months:.1f} months", fontsize=8)

    plt.tight_layout()

    # Colorbar
    if cbar:
        make_cbar(dataset[variable], ax, im)


    return im


def plot_depths(dataset, variable, time, eta=None, xi=None, max_depth=500, title=None, levels=None, override_title=False, gen_levels=False):
    """
    Plots depth profiles of a given variable at multiple time indices.

    Generates a grid of subplots, each showing the depth profile at a different time index.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        variable (str): The name of the variable to plot.
        time (int or list of int): Time index or list of time indices to plot.
        eta (int, optional): Index along latitudinal dimension (for zonal depth profile). Defaults to None (meridional profile).
        xi (int, optional): Index along longitudinal dimension (for meridional depth profile). Defaults to None (zonal profile).
        max_depth (float, optional): Maximum depth to plot. Defaults to 500m.
        title (str, optional): Overall title for the set of plots. Defaults to None (auto-generated).
        levels (list of float, optional): Contour levels to overlay on all depth plots. Defaults to None (no contours).
        override_title (bool, optional): If True, use the provided 'title' directly as the figure suptitle. Defaults to False (appends variable name).
        gen_levels (bool, optional): Whether to automatically generate contour levels for all depth plots. Defaults to False.
    """
    # Ensure time is a list
    time = [time] if not isinstance(time, list) else time

    # Create the figure and axes for multiple subplots
    fig, axs = create_subplots(len(time))

    # Determine global vmin and vmax
    vmin = dataset[variable][time].min().values
    vmax = dataset[variable][time].max().values

    # Plot each time index using plot_depth
    for i, idx in enumerate(time):
        im = plot_depth(dataset, variable, idx, eta, xi, max_depth, axs[i], title=f"{dataset.time.values[idx]/(3600*24*30):.1f} months", cbar=False, levels=levels, vmin=vmin, vmax=vmax, gen_levels=gen_levels)

        # Set axis to outer labels only
        axs[i].label_outer()

    # Title handling for the figure
    title = title if override_title else (f"{dataset[variable].attrs['long_name']}" if title is None else f"{title} ({dataset[variable].attrs['long_name']})")
    fig.suptitle(title, fontsize=16)

    # Apply tight layout for better spacing
    plt.tight_layout()

    # Add a colorbar to the figure
    cbar = fig.colorbar(im, ax=axs.ravel().tolist())
    cbar.set_label(f"{dataset[variable].attrs['long_name']} ({dataset[variable].attrs['units']})")


def plot_surface_vars(dataset, time, title=None):
    """
    Plots surface maps for a set of standard physical variables.

    Variables plotted are: 'u', 'v', 'w', 'temp', 'salt', 'rho', 'KE', 'RV', 'zeta'.
    Generates a grid of subplots for these variables at a given time index.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        time (int): The time index to plot.
        title (str, optional): Overall title for the set of plots. Defaults to None (auto-generated).

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object.
            - ax (list of matplotlib.axes.Axes): A list of axes objects for the generated subplots.
    """

    vars = ["u", "v", "w", "temp", "salt", "rho", "KE", "RV", "zeta"]
    fig, ax = create_subplots(len(vars))

    for i, var in enumerate(vars):
        plot_surface(dataset, var, time, ax=ax.flat[i])
        ax[i].label_outer()

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()

    return fig, ax


def plot_depth_vars(dataset, time, eta=None, xi=None, max_depth=500, title=None):
    """
    Plots depth profiles for a set of standard physical variables.

    Variables plotted are: 'u', 'v', 'w', 'temp', 'salt', 'rho', 'KE', 'RV'.
    Generates a grid of subplots for these variables at a given time index, either zonal or meridional profiles.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        time (int): The time index to plot.
        eta (int, optional): Index along latitudinal dimension (for zonal depth profile). Defaults to None (meridional profile).
        xi (int, optional): Index along longitudinal dimension (for meridional depth profile). Defaults to None (zonal profile).
        max_depth (float, optional): Maximum depth to plot. Defaults to 500m.
        title (str, optional): Overall title for the set of plots. Defaults to None (auto-generated).

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object.
            - ax (list of matplotlib.axes.Axes): A list of axes objects for the generated subplots.
    """

    vars = ["u", "v", "w", "temp", "salt", "rho", "KE", "RV"]
    fig, ax = create_subplots(len(vars))

    for i, var in enumerate(vars):
        plot_depth(dataset, var, time, eta=eta, xi=xi, ax=ax.flat[i], max_depth=max_depth)

    #manually turn off inner labels
    for i in [1, 2, 3, 5, 6, 7]:
        ax[i].set_ylabel('')
        ax[i].set_yticklabels([])
    for i in [0, 1, 2, 3]:
        ax[i].set_xlabel('')
        ax[i].set_xticklabels([])
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()

    return fig, ax


def plot_surface_biovars(dataset, time, title=None, max_columns=2, figsize=None):
    """
    Plots surface maps for a set of standard biological variables.

    Variables plotted are: 'PHYTO', 'NO3'.
    Generates a grid of subplots for these variables at a given time index.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        time (int): The time index to plot.
        title (str, optional): Overall title for the set of plots. Defaults to None (auto-generated).
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to 2.
        figsize (tuple, optional): Figure size (width, height). Defaults to None.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object.
            - ax (list of matplotlib.axes.Axes): A list of axes objects for the generated subplots.
    """

    # vars = ["PHYTO", "CHLA", "NO3"]
    vars = ["PHYTO", "NO3"]

    fig, ax = create_subplots(len(vars), max_columns=max_columns, figsize=figsize)

    for i, var in enumerate(vars):
        plot_surface(dataset, var, time, ax=ax[i])
        ax[i].label_outer()

    plt.suptitle(title, fontsize=15)
    # plt.tight_layout()

    return fig, ax

def plot_depth_biovars(dataset, time, depth=500, eta=None, xi=None, shading="gouraud", max_columns=2, figsize=None):
    """
    Plots depth profiles for a set of standard biological variables.

    Variables plotted are: 'PHYTO', 'NO3'.
    Generates a grid of subplots for these variables at a given time index, either zonal or meridional profiles.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        time (int): The time index to plot.
        depth (float, optional): Maximum depth to plot. Defaults to 500m.
        eta (int, optional): Index along latitudinal dimension (for zonal depth profile). Defaults to None (meridional profile).
        xi (int, optional): Index along longitudinal dimension (for meridional depth profile). Defaults to None (zonal profile).
        shading (str, optional): Shading mode for pcolormesh. Defaults to "gouraud".
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to 2.
        figsize (tuple, optional): Figure size (width, height). Defaults to None.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure): The figure object.
            - ax (list of matplotlib.axes.Axes): A list of axes objects for the generated subplots.
    """
    vars = ["PHYTO", "NO3"]
    fig, ax = create_subplots(len(vars), max_columns=max_columns, figsize=figsize)


    for i, var in enumerate(vars):
        plot_depth(dataset, var, time, max_depth=depth, ax=ax[i], eta=eta, xi=xi, shading=shading)
        ax[i].label_outer()

    # plt.tight_layout()

    return fig, ax

from .computation_utils import compute_average_KE, compute_total_KE # Import here to avoid circular dependency
from .timeseries_plotting_utils import plot_over_time # Import here to avoid circular dependency
from .data_min_max_utils import get_minmax_datasets # Import here to avoid circular dependency
