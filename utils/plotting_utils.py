import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from .colormap_utils import get_colormap_and_limits
from .depth_utils import get_zlevs, get_depth_index
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_surface(dataset, variable, time_idx, ax=None, title=None, cmap=None, vmin=None, vmax=None, cbar=True, time_unit='months', auto_minmax=False):
    """
    Plots a surface map of a given variable from an xarray Dataset.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variable to plot.
        variable (str): The name of the variable to plot.
        time_idx (int): The time index to plot.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. Defaults to None (creates a new figure and axes).
        title (str, optional): The title of the plot. Defaults to None (auto-generated).
        cmap (str, optional): Colormap to use. Defaults to None (auto-determined).
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        cbar (bool, optional): Whether to add a colorbar. Defaults to True.
        time_unit (str, optional): Unit for time display in the title ('hours', 'days', 'weeks', 'months', 'years'). Defaults to 'months'.
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.
    Returns:
        tuple: A tuple containing the axes object and the image object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    data = dataset[variable][time_idx, 0].values
    cmap, vmin, vmax = get_colormap_and_limits(data, vmin=vmin, vmax=vmax)
    if vmin is None or vmax is None:
        cmap, vmin, vmax = get_colormap_and_limits(data, vmin=vmin, vmax=vmax)
    if auto_minmax:
        from .data_min_max_utils import get_minmax_datasets # Import here to avoid circular dependency
        global_min_max = get_minmax_datasets([dataset], variable) # Pass dataset as a list
        vmin, vmax = global_min_max[0], global_min_max[1] # unpack min and max values

    X, Y = dataset.lon_km.values, dataset.lat_km.values
    time_in_months = dataset.time.values[time_idx] / (3600 * 24 * 30)  # Convert seconds to months
    im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')

    if title is None:
        title = f"{variable} at surface, Time: {time_in_months:.1f} {time_unit}"
    ax.set_title(title)
    ax.set_xlabel("Longitude (km)")
    ax.set_ylabel("Latitude (km)")
    ax.set_aspect('equal', adjustable='box')

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label=variable)

    return ax, im


def plot_surfaces(dataset, variable, time_idx=slice(0, 5), titles=None, cmap=None, vmin=None, vmax=None, suptitle=None, time_unit='months', auto_minmax=False):
    """
    Plots a series of surface maps for a given variable over multiple time indices.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variable to plot.
        variable (str): The name of the variable to plot.
        time_idx (slice, list): Time indices to plot. Defaults to slice(0, 5) (first 5 time steps).
        titles (list of str, optional): Titles for each subplot. Defaults to None (auto-generated titles).
        cmap (str, optional): Colormap to use. Defaults to None (auto-determined).
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        suptitle (str, optional): Overall suptitle for the figure. Defaults to None (auto-generated).
        time_unit (str, optional): Unit for time display in the title ('hours', 'days', 'weeks', 'months', 'years'). Defaults to 'months'.
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.
    Returns:
        tuple: A tuple containing the figure object and the axes objects.
    """
    if isinstance(time_idx, int):  # Handle single time index input
        time_idx = [time_idx]
    num_plots = len(time_idx)
    fig, axs = create_subplots(num_plots)


    if vmin is None or vmax is None:
        all_data = np.concatenate([dataset[variable][t_idx].values.flatten() for t_idx in time_idx])
        cmap, vmin, vmax = get_colormap_and_limits(all_data, vmin=vmin, vmax=vmax)
    if auto_minmax:
        from .data_min_max_utils import get_minmax_datasets # Import here to avoid circular dependency
        global_min_max = get_minmax_datasets([dataset], variable, time=time_idx) # Pass dataset as list and time slice
        vmin, vmax = global_min_max[0], global_min_max[1] # unpack min and max values

    ims = []
    i = 0
    for t_idx in time_idx:
        time_in_months = dataset.time.values[t_idx] / (3600 * 24 * 30)
        if titles is None:
            title = f"Time: {time_in_months:.1f} months"
        ax, im = plot_surface(dataset, variable, t_idx, ax=axs[i], title=title, vmin=vmin, vmax=vmax, cbar=False, time_unit=time_unit, auto_minmax=False) # auto_minmax is already handled at surfaces level
        ims.append(im)
        i += 1

    if suptitle is None:
        suptitle = f"Surface plots of {variable}"
    fig.suptitle(suptitle)
    add_colorbar_to_subplots(fig, axs, ims, variable)  # Pass ims to use the last image for colorbar levels
    return fig, axs


def plot_depth(dataset, variable, time, eta, xi, max_depth=500, ax=None, title=None, cmap=None, vmin=None, vmax=None, cbar=True, shading='gouraud', auto_minmax=False):
    """
    Plots a depth profile map of a given variable from an xarray Dataset at a specific time and location (eta, xi).

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variable to plot.
        variable (str): The name of the variable to plot.
        time (int): The time index to plot.
        eta (int): The eta index (latitude-like) to plot.
        xi (int): The xi index (longitude-like) to plot.
        max_depth (int, optional): Maximum depth to consider for plotting. Defaults to 500m.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. Defaults to None (creates a new figure and axes).
        title (str, optional): The title of the plot. Defaults to None (auto-generated).
        cmap (str, optional): Colormap to use. Defaults to None (auto-determined).
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        cbar (bool, optional): Whether to add a colorbar. Defaults to True.
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.

    Returns:
        matplotlib.image.QuadMesh: The image object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    z_rho = dataset.z_rho[time, :, eta, xi].values
    z_w = dataset.z_w[time, :, eta, xi].values
    variable_data = dataset[variable][time, :, eta, xi].values

    z_idx = get_depth_index(z_rho, max_depth)

    Z = -z_rho[z_idx:]
    data = variable_data[z_idx:]

    cmap, vmin, vmax = get_colormap_and_limits(data, vmin=vmin, vmax=vmax)
    cmap = cmap if cmap is not None else 'viridis'
    if auto_minmax:
        from .data_min_max_utils import get_minmax_datasets # Import here to avoid circular dependency
        global_min_max = get_minmax_datasets([dataset], variable, time=time, max_depth=max_depth) # Pass dataset as list and time and depth
        vmin, vmax = global_min_max[0], global_min_max[1] # unpack min and max values

    im = ax.pcolormesh(dataset.ocean_time.values[[time]], Z, data, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)

    # Labels and formatting
    time_in_months = dataset.time.values[time] / (3600 * 24 * 30)
    if title is None:
        title = f"{variable} depth profile, Time: {time_in_months:.1f} months, Location: eta={eta}, xi={xi}"
    ax.set_title(title)
    ax.set_ylabel("Depth (m)")
    ax.set_xticks([])  # remove xticks for time axis
    ax.invert_yaxis()  # Invert y-axis to have depth increasing downwards

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cax, label=variable)

    return im


def plot_depths(dataset, variable, time_indices=slice(0, 5), eta=50, xi=50, max_depth=500, titles=None, cmap=None, vmin=None, vmax=None, suptitle=None, cbar=True, auto_minmax=False):
    """
    Plots a series of depth profile maps for a given variable over multiple time indices at a fixed location (eta, xi).

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variable to plot.
        variable (str): The name of the variable to plot.
        time_indices (slice, list): Time indices to plot. Defaults to slice(0, 5) (first 5 time steps).
        eta (int): The eta index (latitude-like) to plot. Defaults to 50.
        xi (int): The xi index (longitude-like) to plot. Defaults to 50.
        max_depth (int, optional): Maximum depth to consider for plotting. Defaults to 500m.
        titles (list of str, optional): Titles for each subplot. Defaults to None (auto-generated titles).
        cmap (str, optional): Colormap to use. Defaults to None.
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        suptitle (str, optional): Overall title for the figure. Defaults to None.
        cbar (bool, optional): Whether to add a colorbar. Defaults to True.
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.

    Returns:
        tuple: Figure and axes objects.
    """
    if isinstance(time_indices, int):  # Handle single time index input
        time_indices = [time_indices]
    num_plots = len(time_indices)
    fig, axs = create_subplots(num_plots)


    if vmin is None or vmax is None:
        all_data = np.concatenate([dataset[variable][time, :, eta, xi].values.flatten() for time in time_indices])
        cmap, vmin, vmax = get_colormap_and_limits(all_data, vmin=vmin, vmax=vmax)
    if auto_minmax:
        from .data_min_max_utils import get_minmax_datasets # Import here to avoid circular dependency
        global_min_max = get_minmax_datasets([dataset], variable, time=time_indices, max_depth=max_depth) # Pass dataset as list and time slice
        vmin, vmax = global_min_max[0], global_min_max[1] # unpack min and max values

    ims = []
    i = 0
    for t_idx in time_indices:
        time_in_months = dataset.time.values[t_idx] / (3600 * 24 * 30)
        if titles is None:
            title = f"Time: {time_in_months:.1f} months"
        im = plot_depth(dataset, variable, time=t_idx, eta=eta, xi=xi, max_depth=max_depth, ax=axs[i], title=title, vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, auto_minmax=False) # auto_minmax is handled at depths level
        ims.append(im)
        i += 1

    if suptitle is None:
        suptitle = f"Depth profiles of {variable} at eta={eta}, xi={xi}"
    fig.suptitle(suptitle)
    add_colorbar_to_subplots(fig, axs, ims, variable, orientation='vertical')
    return fig, axs


def compare_datasets(datasets, variable, time_idx=0, labels=None, cmap=None, vmin=None, vmax=None, robust=False, auto_minmax=False):
    """
    Compares surface plots of a given variable from multiple xarray Datasets at the same time index.

    Parameters:
        datasets (list of xarray.Dataset): A list of datasets to compare.
        variable (str): The name of the variable to plot.
        time_idx (int): The time index to plot. Defaults to 0 (first time step).
        labels (list of str, optional): Labels for each dataset in the plot titles. Defaults to None (uses dataset index).
        cmap (str, optional): Colormap to use. Defaults to None (viridis).
        vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-determined).
        vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-determined).
        robust (bool, optional): Use robust min/max values for color scaling. Defaults to False.
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.

    Returns:
        tuple: Figure and axes objects.
    """
    num_datasets = len(datasets)
    fig, axs = create_subplots(num_datasets)

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(num_datasets)]

    if vmin is None or vmax is None:
        combined_data = np.concatenate([ds[variable][time_idx].values.flatten() for ds in datasets])
        cmap, vmin, vmax = get_colormap_and_limits(combined_data, vmin=vmin, vmax=vmax)
    if auto_minmax:
        from .data_min_max_utils import get_minmax_datasets # Import here to avoid circular dependency
        global_min_max = get_minmax_datasets(datasets, variable, time=time_idx) # Pass datasets list and time slice
        vmin, vmax = global_min_max[0], global_min_max[1] # unpack min and max values

    if cmap is None:
        cmap = 'viridis'

    ims = []
    for i, dataset in enumerate(datasets):
        ax = axs[i]
        ax_im = plot_surface(dataset, variable, time_idx, ax=ax, title=f"{labels[i]}", vmin=vmin, vmax=vmax, cmap=cmap, cbar=False)[1]  # Get the image object
        ims.append(ax_im)

    fig.suptitle(f"Comparison of {variable} at Time {time_idx}")
    add_colorbar_to_subplots(fig, axs, ims, variable)
    return fig, axs


def create_subplots(num_plots, max_columns=2, figsize=None):
    """
    Creates a grid of subplots dynamically adjusting rows and columns based on the number of plots.

    Parameters:
        num_plots (int): The number of subplots to create.
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to 2.
        figsize (tuple, optional): Figure size. Defaults to None (auto-determined).

    Returns:
        tuple: Figure and axes objects.
    """
    num_rows = (num_plots + max_columns - 1) // max_columns  # Calculate rows needed
    num_cols = min(num_plots, max_columns)  # Adjust columns for the last row

    if figsize is None:
        figsize = (num_cols * 6, num_rows * 5)  # Adjust figure size based on the number of subplots

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)

    if num_plots == 1: # Handle single plot case to ensure axs is still iterable
        axs = np.array([axs]) # make it a 1D array

    # Hide any unused subplots if num_plots is not a perfect grid fill
    if num_plots < num_rows * num_cols:
        for i in range(num_plots, num_rows * num_cols):
            fig.delaxes(axs.flatten()[i]) # remove extra subplots

    return fig, axs


def add_colorbar_to_subplots(fig, axs, ims, variable, orientation='right'):
    """
    Adds a single colorbar to a series of subplots.

    Parameters:
        fig (matplotlib.figure.Figure): The figure object containing the subplots.
        axs (numpy.ndarray): Array of axes objects.
        ims (list): List of image objects from the subplots.
        variable (str): The name of the variable for the colorbar label.
        orientation (str, optional): Orientation of the colorbar ('right' or 'bottom'). Defaults to 'right'.
    """
    if not ims:  # Check if ims is empty
        return

    if orientation == 'right':
        pos = axs[-1].get_position()
        cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.02, pos.height])  # [left, bottom, width, height]
    elif orientation == 'bottom':
        pos = axs[-1].get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.04, pos.width, 0.02])  # [left, bottom, width, height]
        orientation = 'horizontal' # set orientation to horizontal for colorbar

    fig.colorbar(ims[-1], cax=cbar_ax, label=variable, orientation=orientation)


def plot_surface_vars(dataset, vars, time, max_columns=2, title="Surface Plots", figsize=None, auto_minmax=False):
    """
    Plots surface maps for multiple variables from a single dataset at a given time.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variables to plot.
        vars (list of str): A list of variable names to plot.
        time (int): The time index to plot.
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to 2.
        title (str, optional): Overall title for the figure. Defaults to "Surface Plots".
        figsize (tuple, optional): Figure size. Defaults to None (auto-determined).
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.

    Returns:
        tuple: Figure and axes objects.
    """
    fig, ax = create_subplots(len(vars))

    for i, var in enumerate(vars):
        plot_surface(dataset, var, time, ax=ax.flat[i], auto_minmax=auto_minmax) # Pass auto_minmax
        ax[i].label_outer()

    plt.suptitle(title, fontsize=15)
    return fig, ax


def plot_depth_vars(dataset, vars, time, eta=50, xi=50, max_depth=500, shading='gouraud', max_columns=2, title="Depth Plots", auto_minmax=False):
    """
    Plots depth profile maps for multiple variables from a single dataset at a given time and location (eta, xi).

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variables to plot.
        vars (list of str): A list of variable names to plot.
        time (int): The time index to plot.
        eta (int, optional): The eta index (latitude-like) to plot. Defaults to 50.
        xi (int, optional): The xi index (longitude-like) to plot. Defaults to 50.
        max_depth (int, optional): Maximum depth to consider for plotting. Defaults to 500m.
        title (str, optional): Overall title for the figure. Defaults to "Depth Plots".
        shading (str, optional): Shading mode for pcolormesh. Defaults to "gouraud".
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to 2.
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.

    Returns:
        tuple: Figure and axes objects.
    """
    fig, ax = create_subplots(len(vars))

    for i, var in enumerate(vars):
        plot_depth(dataset, var, time, eta=eta, xi=xi, ax=ax[i], max_depth=max_depth, auto_minmax=auto_minmax) # Pass auto_minmax

    #manually turn off inner labels
    for i in [1, 2, 3, 5, 6, 7]:
        if len(ax) > i: # avoid index error if fewer subplots than labels to turn off
            ax.flat[i].label_outer()

    plt.suptitle(title, fontsize=15)
    return fig, ax


def plot_surface_biovars(dataset, time, max_columns=2, figsize=None, title="Surface Biogeochemical Variables", auto_minmax=False):
    """
    Plots surface maps for a predefined set of biogeochemical variables.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variables to plot.
        time (int): The time index to plot.
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to 2.
        figsize (tuple, optional): Figure size. Defaults to None (auto-determined).
        title (str, optional): Overall title for the figure. Defaults to "Surface Biogeochemical Variables".
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.

    Returns:
        tuple: Figure and axes objects.
    """
    bio_vars = ['temp', 'salt', 'chlorophyll', 'oxygen', 'alkalinity', 'phytoplankton']
    vars = [var for var in bio_vars if var in dataset] # plot only available vars
    if not vars:
        print("No biogeochemical variables found in the dataset.")
        return None, None

    fig, ax = create_subplots(len(vars), max_columns=max_columns, figsize=figsize)

    for i, var in enumerate(vars):
        plot_surface(dataset, var, time, ax=ax[i],  auto_minmax=auto_minmax) # Pass auto_minmax
        ax[i].label_outer()

    plt.suptitle(title, fontsize=15)
    return fig, ax


def plot_depth_biovars(dataset, time, eta=50, xi=50, max_depth=500, shading='gouraud', max_columns=2, figsize=None, title="Depth Biogeochemical Variables", auto_minmax=False):
    """
    Plots depth profile maps for a predefined set of biogeochemical variables.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variables to plot.
        time (int): The time index to plot.
        eta (int, optional): The eta index (latitude-like) to plot. Defaults to 50.
        xi (int, optional): The xi index (longitude-like) to plot. Defaults to 50.
        max_depth (int, optional): Maximum depth to consider for plotting. Defaults to 500m.
        shading (str, optional): Shading mode for pcolormesh. Defaults to "gouraud".
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to 2.
        figsize (tuple, optional): Figure size. Defaults to None (auto-determined).
        title (str, optional): Overall title for the figure. Defaults to "Depth Biogeochemical Variables".
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.

    Returns:
        tuple: Figure and axes objects.
    """
    bio_vars = ['temp', 'salt', 'chlorophyll', 'oxygen', 'alkalinity', 'phytoplankton']
    vars = [var for var in bio_vars if var in dataset] # plot only available vars
    if not vars:
        print("No biogeochemical variables found in the dataset.")
        return None, None

    fig, ax = create_subplots(len(vars), max_columns=max_columns, figsize=figsize)

    for i, var in enumerate(vars):
        plot_depth(dataset, var, time, eta=eta, xi=xi, ax=ax[i], max_depth=max_depth, auto_minmax=auto_minmax) # Pass auto_minmax
        ax[i].label_outer()

    plt.suptitle(title, fontsize=15)
    return fig, ax
