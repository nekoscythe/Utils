import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from .helper_plotting_utils import *
from .data_min_max_utils import get_minmax_datasets
from .depth_utils import get_zlevs, get_depth_index
from .data_manipulation_utils import *


def plot_surface(dataset, variable, time_idx, ax=None, title=None, cmap=None, vmin=None, vmax=None, cbar=True, time_unit='days', shading='gouraud'):
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
    Returns:
        tuple: A tuple containing the axes object and the image object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    var_data = dataset[variable][time_idx, -1] if len(dataset[variable].shape) == 4 else dataset[variable][time_idx] # 4D vs 3D data
    data = interpolate_to_rho(var_data)   #Interpolate data to rho grid
    cmap, vmin, vmax = get_colormap_and_limits(data, vmin=vmin, vmax=vmax)
    

    X, Y = dataset.lon_km.values, dataset.lat_km.values
    im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)

    set_plot_labels(ax, dataset[variable], title=title, xlabel="Longitude (km)", ylabel="Latitude (km)", time_unit=time_unit, time_value=dataset.time[time_idx].values)
    ax.set_aspect('equal', adjustable='box')

    if cbar:
        add_colorbar(fig, im, dataset[variable], ax=ax)

    return ax, im


def plot_surfaces(dataset, variable, time_idx=range(5), titles=None, cmap=None, vmin=None, vmax=None, suptitle=None, time_unit='days', max_columns=2):
    """
    Plots a series of surface maps for a given variable over multiple time indices.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variable to plot.
        variable (str): The name of the variable to plot.
        time_idx (slice, list): Time indices to plot. Defaults to range(5) (first 5 time steps).
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
    fig, axs = create_subplots(num_plots, max_columns=max_columns)
    axs = axs.flatten()


    if vmin is None or vmax is None:
        all_data = np.concatenate([dataset[variable][t_idx, -1].values.flatten() for t_idx in time_idx])
        cmap, vmin, vmax = get_colormap_and_limits(all_data, vmin=vmin, vmax=vmax)

    ims = []
    i = 0
    for t_idx in time_idx:
        title=None
        if titles is not None:
            title = titles[i]
        ax, im = plot_surface(dataset, variable, t_idx, ax=axs[i], title=title, vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, time_unit=time_unit)
        ims.append(im)
        i += 1
        
    for ax in axs:
        ax.label_outer()

    

    if suptitle is None:
        suptitle = f"Surface plots of {variable}"
    fig.suptitle(suptitle)
    add_colorbar_to_subplots(fig, axs, ims, variable)  # Pass ims to use the last image for colorbar levels
    return fig, axs


def plot_depth(dataset, variable, time, eta=None, xi=None, max_depth=500, z_levs=None, ax=None, title=None, cmap=None, vmin=None, vmax=None, cbar=True, shading='gouraud', time_unit='days'):
    """
    Plots a depth profile map of a given variable from an xarray Dataset at a specific time and location (eta or xi)

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
        
    # Either eta or xi must be provided, but not both
    if eta is None and xi is None:
        raise ValueError("Either eta or xi must be provided")
    if eta is not None and xi is not None:
        raise ValueError("Only one of eta or xi can be provided")
    
    data, X, xlabel = slice_data(dataset, variable, time, eta, xi)
    Y, z_var = get_zlevs(dataset) if z_levs is None else (z_levs, None)
    if z_var != "s_rho" and z_var != "s_w":
        z_idx = get_depth_index(Y, max_depth)
        data = data[z_idx:]
        Y = -Y[z_idx:]
        
    cmap, vmin, vmax = get_colormap_and_limits(data, vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(X, Y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)

    # Labels and formatting
    xlabel = None
    if xi is not None: #if xi is provided, we are plotting over latitudes
        xlabel = "Latitude (km)"
    else:
        xlabel = "Longitude (km)"
    set_plot_labels(ax, dataset[variable], xlabel=xlabel, ylabel="Depth (m)", title=title, time_value=dataset.time[time].values, time_unit=time_unit)
    ax.invert_yaxis()  # Invert y-axis to have depth increasing downwards

    if cbar:
        add_colorbar(fig, im, dataset[variable], ax=ax)

    return im


def plot_depths(dataset, variable, time_indices=range(5), eta=None, xi=None, max_depth=500, titles=None, cmap=None, vmin=None, vmax=None, suptitle=None, cbar=True, time_unit='days', max_columns=2):
    """
    Plots a series of depth profile maps for a given variable over multiple time indices at a fixed location (eta, xi).

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variable to plot.
        variable (str): The name of the variable to plot.
        time_indices (slice, list): Time indices to plot. Defaults to range(5) (first 5 time steps).
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
    fig, axs = create_subplots(num_plots, max_columns=max_columns)
    axs = axs.flatten()
    
    if vmin is None or vmax is None:
        all_data = np.concatenate([dataset[variable][time, :].values.flatten() for time in time_indices])
        cmap, vmin, vmax = get_colormap_and_limits(all_data, vmin=vmin, vmax=vmax)

    ims = []
    i = 0
    for t_idx in time_indices:
        title = None
        if titles is not None:
            title = titles[i]
        im = plot_depth(dataset, variable, time=t_idx, eta=eta, xi=xi, max_depth=max_depth, ax=axs[i], title=title, vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, time_unit=time_unit)
        ims.append(im)
        axs[i].label_outer()
        i += 1

    if suptitle is None:
        suptitle = f"Depth profiles of {variable} at " + (f"eta={eta}" if xi is None else f"xi={xi}")
    fig.suptitle(suptitle)
    add_colorbar_to_subplots(fig, axs, ims, variable, orientation='vertical')
    return fig, axs


def compare_datasets(datasets, variable, time_idx=0, eta=None, xi=None, labels=None, cmap=None, vmin=None, vmax=None, time_unit='days', max_columns=2):
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
        auto_minmax (bool, optional): Whether to automatically determine vmin and vmax across datasets. Defaults to False.

    Returns:
        tuple: Figure and axes objects.
    """
    num_datasets = len(datasets)
    fig, axs = create_subplots(num_datasets, max_columns=max_columns)
    axs= axs.flatten()

    if labels is None:
        labels = [f"Dataset {i+1}" for i in range(num_datasets)]

    #check if we are plotting surface or depth
    plotting_depth=False
    if eta is not None or xi is not None:
        plotting_depth=True
        # Either eta or xi must be provided, but not both
        if eta is not None and xi is not None:
            raise ValueError("Only one of eta or xi can be provided")
        
        eta = slice(None) if xi is not None else eta
        xi = slice(None) if eta is not None else xi
        
    if plotting_depth:
        combined_data = np.concatenate([ds[variable][time_idx, :, eta, xi].values.flatten() for ds in datasets])
        eta = None if xi is not None else eta
        xi = None if eta is not None else xi #reset to None for plotting
    else:
        combined_data = np.concatenate([ds[variable][time_idx, -1].values.flatten() for ds in datasets]) # Surface data
    
    cmap, vmin, vmax = get_colormap_and_limits(combined_data, vmin=vmin, vmax=vmax)

    ims = []
    for i, dataset in enumerate(datasets):
        ax = axs[i]
        if plotting_depth:
            ax_im = plot_depth(dataset, variable, time_idx, eta=eta, xi=xi, ax=ax, title=f"{labels[i]}", vmin=vmin, vmax=vmax, cmap=cmap, cbar=False)[1]
        else:
            ax_im = plot_surface(dataset, variable, time_idx, ax=ax, title=f"{labels[i]}", vmin=vmin, vmax=vmax, cmap=cmap, cbar=False)[1]  # Get the image object
        ims.append(ax_im)
        ax.label_outer()
        

    time_val = datasets[0].time[time_idx].values
    time_val, time_unit = convert_time(time_val, unit=time_unit)

    fig.suptitle(f"Comparison of {variable} at {time_val:.1f} {time_unit}")
    
    add_colorbar_to_subplots(fig, axs, ims, variable)
    return fig, axs






def plot_surface_vars(dataset, vars, time, max_columns=2, title="Surface Plots", figsize=None, time_unit="days"):
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
    fig, ax = create_subplots(len(vars), max_columns=max_columns, figsize=figsize)

    for i, var in enumerate(vars):
        plot_surface(dataset, var, time, ax=ax.flat[i], time_unit=time_unit)
        ax[i].label_outer()

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    return fig, ax





def plot_depth_vars(dataset, vars, time, eta=None, xi=None, max_depth=500, shading='gouraud', max_columns=2, title="Depth Plots"):
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
    fig, ax = create_subplots(len(vars), max_columns=max_columns)

    for i, var in enumerate(vars):
        plot_depth(dataset, var, time, eta=eta, xi=xi, ax=ax[i], max_depth=max_depth, shading=shading)

    for ax in ax.flatten():
        ax.label_outer()

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    
    return fig, ax


def plot_surface_biovars(dataset, time, max_columns=2, figsize=None, title="Surface Biogeochemical Variables"):
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
    bio_vars = ['PHYTO', 'NO3']
    vars = [var for var in bio_vars if var in dataset] # plot only available vars
    if not vars:
        print("No biogeochemical variables found in the dataset.")
        return None, None

    fig, ax = create_subplots(len(vars), max_columns=max_columns, figsize=figsize)

    for i, var in enumerate(vars):
        plot_surface(dataset, var, time, ax=ax[i]) # Pass auto_minmax
        ax[i].label_outer()

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    return fig, ax


def plot_depth_biovars(dataset, time, eta=None, xi=None, max_depth=500, shading='gouraud', max_columns=2, figsize=None, title="Depth Biogeochemical Variables"):
    """
    Plots depth profile maps for a predefined set of biogeochemical variables.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variables to plot.
        time (int): The time index to plot.
        eta (int, optional): The eta index (latitude-like) to plot.
        xi (int, optional): The xi index (longitude-like) to plot.
        max_depth (int, optional): Maximum depth to consider for plotting. Defaults to 500m.
        shading (str, optional): Shading mode for pcolormesh. Defaults to "gouraud".
        max_columns (int, optional): Maximum number of columns in the subplot grid. Defaults to 2.
        figsize (tuple, optional): Figure size. Defaults to None (auto-determined).
        title (str, optional): Overall title for the figure. Defaults to "Depth Biogeochemical Variables".

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
        plot_depth(dataset, var, time, eta=eta, xi=xi, ax=ax[i], max_depth=max_depth, shading=shading) 
        ax[i].label_outer()

    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    return fig, ax
