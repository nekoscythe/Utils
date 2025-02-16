import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from IPython.display import HTML
from pyproj import Geod
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
##########################
# PREPROCESSING
##########################
"""

def get_savetime_hours(dataset):
    """
    Calculates the time step of the dataset in hours.

    Parameters:
        dataset (xarray.Dataset): The input dataset containing a 'time' coordinate.

    Returns:
        float: The time step in hours.
    """
    time = dataset.time
    diff = time[1] - time[0]
    return diff.values / 3600

def add_KE(dataset):
    """
    Compute the kinetic energy (KE) of the flow and add it to the dataset.

    KE is calculated as 0.5 * (u_mid^2 + v_mid^2), where u_mid and v_mid are the
    horizontal velocity components interpolated to the rho points.

    Parameters:
        dataset (xarray.Dataset): The dataset containing 'u' and 'v' velocity variables.
    """
    u_mid = 0.5 * (dataset.u[..., :-1].values + dataset.u[..., 1:].values) # interpolate u to rho points
    v_mid = 0.5 * (dataset.v[...,:-1,:].values + dataset.v[..., 1:,:].values) # interpolate v to rho points

    KE = 0.5 * (u_mid**2 + v_mid**2) #this is just 1/2 * velocity^2
    #add KE to dataset
    dataset['KE'] = (('time', 's_rho', 'eta_rho', 'xi_rho'), KE, {'long_name' : 'kinetic energy', 'units': 'meter2 second-2'})
    dataset['KE'] = dataset.KE.chunk()

def add_RV(dataset):
    """
    Compute the relative vorticity (RV) of the flow and add it to the dataset.

    RV is calculated as dv/dx - du/dy, where u and v are horizontal velocity components
    and derivatives are approximated using finite differences on the grid.

    Parameters:
        dataset (xarray.Dataset): The dataset containing 'u', 'v', 'pm', and 'pn' variables.
    """
    pm = dataset.pm.values #1/dx
    pn = dataset.pn.values #1/dy


    pm_psi = 0.25 * (pm[:-1, :-1] + pm[1:, :-1] + pm[:-1, 1:] + pm[1:, 1:]) # interpolate pm to psi points
    pn_psi = 0.25 * (pn[:-1, :-1] + pn[1:, :-1] + pn[:-1, 1:] + pn[1:, 1:]) # interpolate pn to psi points

    dv = dataset.v[:,:,:, 1:].values - dataset.v[:,:,:, :-1].values
    du = dataset.u[:,:, 1:, :].values - dataset.u[:,:, :-1, :].values

    zeta_vort = (dv * pm_psi) - (du * pn_psi) # dv/dx - du/dy

    #add RV to dataset
    dataset['RV'] = (('time', 's_rho', 'eta_v', 'xi_u'), zeta_vort, {'long_name' : 'relative vorticity', 'units': 'second-1'})
    #make chunk size the same as the original dataset
    #1 chunks in 3 graph layers
    dataset['RV'] = dataset.RV.chunk()

def add_dV(dataset):
    """
    Compute the volume (dV) of each cell in the grid and add it to the dataset.

    dV is calculated as dA * dz, where dA is the surface area of the cell and dz is the
    vertical thickness.

    Parameters:
        dataset (xarray.Dataset): The dataset containing 'dA' and 'z_w' variables.
    """
    dz = np.diff(dataset.z_w, axis=1)

    # dA = dx_expanded * dy_expanded  # surface area of each cell
    dV = dataset.dA.values * dz  # volume of each cell

    dataset['dV'] = (('time', 's_rho', 'eta', 'xi'), dV, {'long_name' : 'volume of cells on RHO grid' , 'units': 'meter3'})

def reset_time(dataset):
    """
    Resets the time coordinate in the dataset to start from zero.

    Parameters:
        dataset (xarray.Dataset): The input dataset with a 'time' coordinate.

    Returns:
        xarray.Dataset: The dataset with the time coordinate reset.
    """
    dataset['time'] = dataset.time - dataset.time[0]
    return dataset

def preprocess(dataset, bio=False):
    """
    Applies a series of preprocessing steps to the input dataset.

    The preprocessing steps include:
        - Adding Relative Vorticity (RV)
        - Removing ghost points
        - Adding Kinetic Energy (KE)
        - Adding cell Volume (dV)
        - Adding kilometer grid (lon_km, lat_km)
        - (Optionally) Ensuring non-negative values for biological variables if bio=True
        - Resetting time to start from zero

    Parameters:
        dataset (xarray.Dataset): The input dataset.
        bio (bool, optional): If True, apply non-negativity to biological variables. Defaults to False.

    Returns:
        xarray.Dataset: The preprocessed dataset.
    """
    add_RV(dataset) # we need to compute this before removing ghost points as we need the psi points
    dataset = remove_ghost_points(dataset)
    add_KE(dataset)
    add_dV(dataset)
    add_km_grid(dataset)
    if bio:
        non_negative_bio(dataset)
    dataset = reset_time(dataset)
    return dataset

"""
###########################
# FORCING and INITIAL CONDITIONS EDITING FUNCTIONS^
###########################
"""

def convertToSin(data, field, amplitude):
    """
    Converts a specified field in the dataset to a sinusoidal profile along the Y-dimension.

    The function replaces the values of the given field with a sine wave that varies along
    the Y-dimension (eta). The amplitude and spatial scale of the sine wave are controlled
    by the 'amplitude' parameter and the domain size.

    Parameters:
        data (xarray.Dataset): The dataset to modify.
        field (str): The name of the field to convert to a sine wave.
        amplitude (float): The amplitude of the sine wave.

    Returns:
        xarray.Dataset: A copy of the dataset with the specified field modified.
    """
    data_copy = data.copy()
    one_level = data[field].values[0][:][0]
    Y_field = data[field].dims[1]
    X_field = data[field].dims[2]
    time_field = data[field].dims[0]
    Y = data[Y_field]
    X = data[X_field]
    time = data[time_field]

    L = Y[-1].values - Y[0].values

    new_level = [amplitude * np.sin(2 * np.pi / L * (Y[i].values - Y[0].values)) for i in range(len(Y))]

    new_values = data[field].values
    for t in range(len(time.values)):
        for xi in range(len(X.values)):
            new_values[t, :, xi] = new_level

    data_copy[field].values = new_values

    return data_copy



def linear_gradient(data, field, south_value, north_value):
    """
    Sets a linear gradient for a specified field along the Y-dimension.

    The function modifies the values of the given field to create a linear gradient
    from 'south_value' to 'north_value' along the Y-dimension (eta).

    Parameters:
        data (xarray.Dataset): The dataset to modify.
        field (str): The name of the field to apply the linear gradient to.
        south_value (float): The value at the southern boundary (minimum Y).
        north_value (float): The value at the northern boundary (maximum Y).

    Returns:
        xarray.Dataset: A copy of the dataset with the specified field modified.
    """
    data_copy = data.copy()
    one_level = data[field].values[0][:][0]
    Y_field = data[field].dims[1]
    X_field = data[field].dims[2]
    time_field = data[field].dims[0]
    Y = data[Y_field]
    X = data[X_field]
    time = data[time_field]

    L = Y[-1].values - Y[0].values

    new_values = data[field].values
    for t in range(len(time.values)):
        for xi in range(len(X.values)):
            new_values[t, :, xi] = np.linspace(south_value, north_value, len(Y))

    data_copy[field].values = new_values

    return data_copy


"""
###########################
COMPUTATION FUNCTIONS
###########################
"""


#compute surface weighted average
def compute_surface_average(ds, var, s_rho=-1):
    """
    Computes the surface weighted average of a variable.

    The average is weighted by the surface area (dA) of each grid cell.

    Parameters:
        ds (xarray.Dataset): The dataset containing the variable and 'dA'.
        var (str): The name of the variable to average.
        s_rho (int, optional): The s_rho level to consider as the surface. Defaults to -1 (top level).

    Returns:
        np.ndarray: The time series of the surface weighted average.
    """
    data = ds[var][:, s_rho]
    dA = ds.dA.values
    dA = dA.reshape((1, dA.shape[0], dA.shape[1]))
    weighted = data * dA
    return np.mean(weighted, axis=(1, 2)) / np.mean(dA)


def compute_weighted_average(dataset, variable, depth=-1):
    """
    Calculate the weighted average of a given variable over time and volume.

    The average is weighted by the volume (dV) of each grid cell. Optionally, the average
    can be computed for depths shallower than a specified 'depth'.

    Parameters:
        dataset (xarray.Dataset): The dataset containing the variable and 'dV', 'z_rho'.
        variable (str): The name of the variable to calculate the weighted average for.
        depth (int, optional): The depth in meters to calculate the weighted average for.
                               Defaults to -1, which means average over all depths.

    Returns:
        np.ndarray: The time series of the volume-weighted average of the variable.
    """
    data_vals = dataset[variable].values
    dV = dataset.dV.values

    if depth != -1:
        z_levs , _ = get_zlevs(dataset)
        idx = get_depth_index(z_levs, depth)
        data_vals = dataset[variable][:, idx:].values
        dV = dataset.dV[:, idx:].values

    total_volume = np.sum(dV, axis=(1, 2, 3))
    weighted_avg_over_time = np.sum(data_vals * dV, axis=(1, 2, 3)) / total_volume

    return weighted_avg_over_time


def compute_total_KE(dataset):
    """
    Compute the total kinetic energy (KE) of the flow over time.

    Total KE is calculated by summing the KE in each grid cell, weighted by the cell volume
    and density.

    Parameters:
        dataset (xarray.Dataset): The dataset containing 'KE', 'dV', 'rho', and 'rho0'.

    Returns:
        np.ndarray: The time series of the total kinetic energy of the flow.
    """

    dV = dataset.dV.values # volume of cell centered at rho points
    densities = dataset.rho.values + dataset.rho0.values # remove ghost cells and add rho0
    total_KE = dataset.KE.values * dV * densities # multiply by volume and density to get kinetic energy

    total_KE_over_time = np.sum(total_KE, axis=(1, 2, 3))

    return total_KE_over_time


def compute_average_KE(dataset):
    """
    Compute the average kinetic energy (KE) of the flow over time.

    Average KE is calculated as the volume-weighted average of KE over all grid cells.

    Parameters:
        dataset (xarray.Dataset): The dataset containing 'KE' and 'dV'.

    Returns:
        np.ndarray: The time series of the average kinetic energy of the flow.
    """
    #weigted average of KE
    total_volume = np.sum(dataset.dV.values, axis=(1, 2, 3))

    average_KE_over_time = np.sum(dataset.KE.values * dataset.dV.values, axis=(1, 2, 3)) / total_volume

    return average_KE_over_time


"""
###########################
HELPER FUNCTIONS
###########################
"""

##################
# Plotting Helpers
##################


def get_optimal_subplot_dims(n, max_columns=4):
    """
    Determine the optimal subplot dimensions (rows, columns) for a given number of plots.

    The function aims to arrange subplots in a grid that is as close to square as possible,
    with a maximum number of columns specified by 'max_columns'.

    Parameters:
        n (int): The number of subplots to arrange.
        max_columns (int, optional): The maximum number of columns in the subplot grid. Defaults to 4.

    Returns:
        tuple: A tuple (rows, columns) representing the optimal subplot dimensions.
    """
    # Function to find factors of a number
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))

    # Find the optimal factor pair (rows, columns)
    best_rows, best_cols = None, None
    for rows, cols in factors:
        if cols <= max_columns:
            if best_rows is None or abs(rows - cols) < abs(best_rows - best_cols):
                best_rows, best_cols = rows, cols

    # If no optimal pair found, we need to ensure columns don't exceed the max
    if best_rows is None or best_cols > max_columns:
        # Use default behavior if no suitable pair found (e.g., columns = max_columns)
        best_cols = min(n, max_columns)
        best_rows = (n + best_cols - 1) // best_cols

    return best_rows, best_cols

def create_subplots(n, max_columns=4, figsize=None):
    """
    Creates a figure and a set of subplots arranged in an optimal grid.

    Uses 'get_optimal_subplot_dims' to determine the subplot layout.

    Parameters:
        n (int): The number of subplots to create.
        max_columns (int, optional): The maximum number of columns in the subplot grid. Defaults to 4.
        figsize (tuple, optional): The figure size (width, height) in inches. Defaults to None,
                                    which uses a default size based on the number of columns and rows.

    Returns:
        tuple: A tuple (fig, axes) where 'fig' is the matplotlib Figure object and 'axes'
               is a list of matplotlib Axes objects (flattened if more than one subplot).
    """
    rows, cols = get_optimal_subplot_dims(n, max_columns)

    if figsize is not None:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4)) # 4x4 aspect ratio + 1 for colorbar

    # Flatten the axes if necessary
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Remove any extra axes if there are fewer than n subplots
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    return fig, axes

def get_colormap_and_limits(data, vmin=None, vmax=None):
    """
    Determine colormap and value limits for plotting based on the data range.

    If the data is entirely positive, 'turbo' colormap is used. Otherwise, 'seismic'
    colormap is used to represent both positive and negative values.

    Parameters:
        data (np.ndarray): The data to be plotted.
        vmin (float, optional): Minimum value for the colormap. If None, it's determined from data. Defaults to None.
        vmax (float, optional): Maximum value for the colormap. If None, it's determined from data. Defaults to None.

    Returns:
        tuple: A tuple (cmap, vmin, vmax) containing the colormap name, minimum value, and maximum value.
    """
    pos_data = (data >= 0).all()  # Improved readability
    cmap = "turbo" if pos_data else "seismic"
    vmax = np.nanmax(np.abs(data)) if vmax is None else vmax
    vmin = np.nanmin(data) if vmin is None else vmin
    return cmap, vmin, vmax



def make_cbar(var_data, ax, im):
    """
    Adds a colorbar to a plot.

    Parameters:
        var_data (xarray.DataArray): The data variable being plotted (used for units).
        ax (matplotlib.axes.Axes): The axes object to which the colorbar should be added.
        im (matplotlib.collections.QuadMesh): The image object returned by pcolormesh or similar.

    Returns:
        matplotlib.colorbar.Colorbar: The colorbar object.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    cbar = ax.get_figure().colorbar(im, cax=cax)
    cbar.set_label(f"{var_data.attrs['units']}")
    return cbar



######################
# Data Manipulation
######################

def scale_depth(z_levs, z_crit, max_multiplier=1, poly_order=1):
    """
    Scales depth levels based on a critical depth.

    This function applies a scaling factor to depth levels ('z_levs') based on a critical depth
    ('z_crit'). Depths shallower than 'z_crit' are scaled by 'max_multiplier', while deeper
    levels are scaled using a polynomial function to smoothly transition the scaling.

    Parameters:
        z_levs (np.ndarray): Array of depth levels (negative values, increasing downwards).
        z_crit (float): The critical depth (negative value) below which scaling starts to decrease.
        max_multiplier (float, optional): The maximum scaling multiplier applied to shallow depths. Defaults to 1.
        poly_order (int, optional): The order of the polynomial used for scaling deeper depths. Defaults to 1.

    Returns:
        np.ndarray: Array of scaling factors for each depth level.
    """
    z_crit = -1 * z_crit
    z_crit = max(z_levs.min(), z_crit)  # Ensure z_crit is within the range of z_levs

    # Compute scaling factor based on a power curve
    above_crit = (z_levs < z_crit) * max_multiplier
    below_crit = (z_levs >= z_crit) * (1 - (z_crit - z_levs) / z_crit)**poly_order * max_multiplier  # Cubic for sharper acceleration

    return below_crit + above_crit


def get_minmax_datasets(datasets, field, time=None, s_rho=None, eta=None, xi=None, max_depth=500, z_levs=None):
    """
    Determine the global minimum and maximum values for a field across multiple datasets.

    This function iterates through a list of datasets and finds the overall minimum and maximum
    values of a specified field, considering optional time, depth, and spatial slices.

    Parameters:
        datasets (list of xarray.Dataset): A list of datasets to compare.
        field (str): The name of the field to find min/max values for.
        time (int, list, slice, optional): Time index or slice to consider. Defaults to None (all time steps).
        s_rho (int, list, slice, optional): s_rho index or slice to consider. Defaults to None (all s_rho levels).
        eta (int, list, slice, optional): eta index or slice to consider. Defaults to None (all eta points).
        xi (int, list, slice, optional): xi index or slice to consider. Defaults to None (all xi points).
        max_depth (float, optional): Maximum depth to consider (used if s_rho is None). Defaults to 500m.
        z_levs (np.ndarray, optional): Depth levels array (if available, to avoid re-calculation). Defaults to None.

    Returns:
        np.ndarray: An array [global_min, global_max] containing the global minimum and maximum values.
    """
    # Use slices to handle None values
    time = slice(None) if time is None else time
    s_rho = slice(None) if s_rho is None else s_rho
    eta = slice(None) if eta is None else eta
    xi = slice(None) if xi is None else xi
    if max_depth is not None and s_rho == slice(None):
        z_levs = datasets[0].z_rho[0,:,0,0].values if z_levs is None else z_levs
        z_idx = get_depth_index(z_levs, max_depth)
        s_rho = slice(z_idx, None)

    # Collect all data arrays and compute min/max in one step
    data = np.array([ds[field][time, s_rho, eta, xi].values for ds in datasets])
    #check if data is non-negative
    if (data < 0).any():
        global_max = np.abs(data).max()
        global_min = -global_max
    else:
        global_min = data.min()
        global_max = data.max()


    return np.array([global_min, global_max])


def print_minmax_biovars(ds):
    """
    Prints the minimum and maximum values of biological variables (PHYTO, NO3, CHLA) in a dataset.

    This function is primarily for quick inspection of the range of biological variables.

    Parameters:
        ds (xarray.Dataset): The dataset to inspect.
    """
    min_phyto, max_phyto = ds.PHYTO.min().values, ds.PHYTO.max().values
    min_no3, max_no3 = ds.NO3.min().values, ds.NO3.max().values
    min_chla, max_chla = ds.CHLA.min().values, ds.CHLA.max().values

    print(f"Min Phyto: {min_phyto:.2f}, Max Phyto: {max_phyto:.2f}, "
          f"Min NO3: {min_no3:.2f}, Max NO3: {max_no3:.2f}, "
          f"Min CHLA: {min_chla:.2f}, Max CHLA: {max_chla:.2f}")


def plot_all(ds):
    """
    Generates and displays surface and depth plots for biological variables over all time steps.

    This function iterates through each time step in the dataset and calls 'plot_surface_biovars'
    and 'plot_depth_biovars' to create plots for PHYTO, NO3, and CHLA. Plots are displayed using plt.show().

    Parameters:
        ds (xarray.Dataset): The dataset to plot.
    """
    for time in np.arange(0, len(ds.time), 1):
       plot_surface_biovars(ds, time)
       plot_depth_biovars(ds, time)
       plt.show()



def non_negative_bio(ds):
    """
    Clips biological variables (CHLA, NO3, PHYTO) to ensure non-negative values.

    This function modifies the dataset in-place, setting any negative values in 'CHLA', 'NO3',
    and 'PHYTO' variables to zero.

    Parameters:
        ds (xarray.Dataset): The dataset to modify.
    """
    ds['CHLA'] = ds.CHLA.clip(min=0)
    ds['NO3'] = ds.NO3.clip(min=0)
    ds['PHYTO'] = ds.PHYTO.clip(min=0)



def combine_along_time(datasets):
    """
    Combines two datasets along the time dimension, removing overlapping time steps.

    This function is designed to concatenate two datasets that represent consecutive time periods.
    It removes the last time step of the first dataset to avoid overlap with the second dataset.

    Parameters:
        datasets (list of xarray.Dataset): A list containing two datasets to combine.

    Returns:
        xarray.Dataset: The combined dataset.
    """
    static_vars = [var for var in datasets[0].data_vars if "time" not in datasets[0][var].dims]
    dynamic_vars = [var for var in datasets[0].data_vars if "time" in datasets[0][var].dims]

    # Filter out the last time step of the first dataset
    first_dataset_filtered = datasets[0].sel(time=slice(None, datasets[0].time[-2]))

    # Extract dynamic data
    dynamic_data = [first_dataset_filtered[dynamic_vars], datasets[1][dynamic_vars]]
    combined_dynamic = xr.combine_by_coords(dynamic_data, combine_attrs="override")

    # Use the static data from the first dataset
    static_data = datasets[0][static_vars]

    # Merge static and combined dynamic data
    combined_dataset = xr.merge([static_data, combined_dynamic])

    return combined_dataset


def lon_to_km(latitude, longitudes):
    """
    Converts an array of longitudes to kilometers, assuming a constant latitude.

    Calculates distances eastwards from the first longitude in the array.

    Parameters:
        latitude (float): The latitude at which to calculate distances.
        longitudes (np.ndarray): Array of longitude values.

    Returns:
        np.ndarray: Array of distances in kilometers corresponding to the longitudes.
    """
    # Define geodetic calculator (WGS84 ellipsoid)
    geod = Geod(ellps="WGS84")

    # Calculate the distance along latitude (north-south)
    x_km = np.zeros(len(longitudes))
    for i in range(1, len(longitudes)):
        _, _, dist = geod.inv(longitudes[i-1], latitude, longitudes[i], latitude)
        x_km[i] = x_km[i-1] + dist / 1000

    return x_km

def lat_to_km(latitudes, longitude):
    """
    Converts an array of latitudes to kilometers, assuming a constant longitude.

    Calculates distances northwards from the first latitude in the array.

    Parameters:
        latitudes (np.ndarray): Array of latitude values.
        longitude (float): The longitude at which to calculate distances.

    Returns:
        np.ndarray: Array of distances in kilometers corresponding to the latitudes.
    """
    # Define geodetic calculator (WGS84 ellipsoid)
    geod = Geod(ellps="WGS84")

    # Calculate the distance along longitude (east-west)
    y_km = np.zeros(len(latitudes))
    for j in range(1, len(latitudes)):
        _, _, dist = geod.inv(longitude, latitudes[j-1], longitude, latitudes[j])
        y_km[j] = y_km[j-1] + dist / 1000

    return y_km

def add_km_grid(dataset):
    """
    Adds kilometer-based longitude and latitude coordinates to the dataset.

    Creates 'lon_km' and 'lat_km' coordinates based on 'lon_rho' and 'lat_rho',
    representing distances in kilometers from the western and southern boundaries, respectively.

    Parameters:
        dataset (xarray.Dataset): The dataset to which kilometer grids should be added.
    """
    latitudes = dataset.lat_rho[:, 0].values
    longitudes = dataset.lon_rho[0, :].values

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError("latitudes and longitudes must be 1D arrays.")

    # Calculate the distance along latitude (north-south)
    y_km = lat_to_km(latitudes, longitudes[0])

    # Calculate the distance along longitude (east-west)
    x_km = lon_to_km(latitudes[0], longitudes)

    # Create 2D arrays for the x and y distances
    X_km, Y_km = np.meshgrid(x_km, y_km)

    #Add meter grid to dataset
    dataset['lon_km'] = (('eta', 'xi'), X_km, {'long_name' : 'X-distance from Western boundary', 'units': 'kilometer'})
    dataset['lat_km'] = (('eta', 'xi'), Y_km, {'long_name' : 'Y-distance from Southern boundary', 'units': 'kilometer'})


def interpolate_to_rho(var_data):
    """
    Interpolates a variable to the rho grid points.

    This function performs linear interpolation to move variable data from its original grid
    (e.g., u, v grids) to the rho grid, which is often the center grid for many ROMS variables.

    Parameters:
        var_data (xarray.DataArray): The variable data to interpolate.

    Returns:
        np.ndarray: The interpolated data on the rho grid.
    """
    #check if the variable is already on the rho points
    data = var_data.values
    if var_data.dims[-1] != 'xi_rho':
        data = (data[...,  :-1] + data[...,  1:]) / 2
    if var_data.dims[-2] != 'eta_rho':
        data = (data[..., :-1, :] + data[..., 1:, :]) / 2
    if len(var_data.dims) <= 2: # 2D data
        return data
    if var_data.dims[-3] != 's_rho':
        if var_data.dims[-1] != 'time': #ignore time dimension
            data = (data[...,  :-1, :, :] + data[...,  1:, :, :]) / 2

    return data

def remove_ghost_points(dataset):
    """
    Removes ghost points from the eta_rho and xi_rho dimensions of the dataset.

    Ghost points are boundary points often added in numerical simulations for boundary conditions.
    This function removes the first and last index along the eta_rho and xi_rho dimensions.

    Parameters:
        dataset (xarray.Dataset): The dataset to remove ghost points from.

    Returns:
        xarray.Dataset: The dataset with ghost points removed.
    """
    return dataset.isel(eta_rho=slice(1,-1), xi_rho=slice(1,-1))

def is_data_positive(data):
    """
    Checks if all values in a numpy array are positive or zero.

    Parameters:
        data (np.ndarray): The input data array.

    Returns:
        bool: True if all values are positive or zero, False otherwise.
    """
    return np.all(data >= 0)



def slice_data(dataset, variable, time_idx, eta=None, xi=None):
    """
    Extracts a depth slice from the dataset along either the eta or xi axis.

    This function selects data for a given variable at a specific time index and then slices
    it along either the latitudinal (eta) or longitudinal (xi) dimension to create a depth profile.

    Parameters:
        dataset (xarray.Dataset): Input dataset.
        variable (str): Variable to extract.
        time_idx (int): Time index for slicing.
        eta (int, optional): Index along latitudinal dimension (for zonal slice). Defaults to None.
        xi (int, optional): Index along longitudinal dimension (for meridional slice). Defaults to None.

    Returns:
        tuple: A tuple containing:
            - data (np.ndarray): Extracted data slice (depth profile).
            - X (np.ndarray): Corresponding coordinate values (lon_km or lat_km, or lon_rho/lat_rho if km coords not available).
            - xlabel (str): Label for the X-axis ("Longitude (km)" or "Latitude (km)").

    Raises:
        ValueError: If both or neither of eta and xi are provided.
    """
    if (eta is not None) == (xi is not None):  # Simplified condition
        raise ValueError("Select either X or Y, but not both.")


    selected_data = dataset[variable].isel(time=time_idx)
    data = interpolate_to_rho(selected_data) # interpolate to rho points

    if eta is not None:
        data = data[:, eta]
        if "lon_km" in dataset:
            X = dataset.lon_km[eta]
        else:
            X = dataset.lon_rho[eta]
        xlabel = "Longitude (km)"
    else:
        data = data[..., :, xi]
        if "lat_km" in dataset:
            X = dataset.lat_km[:, xi]
        else:
            X = dataset.lat_rho[:, xi]
        xlabel = "Latitude (km)"

    return data, X, xlabel




def get_zlevs(dataset):
    """
    Identifies and returns the z-level variable and its values from the dataset.

    This function checks for common z-level variable names ('z_rho', 'z_w', 'z_rho_u', 'z_rho_v')
    in the dataset and returns the first one found, along with its values for the first time step
    and at the corner grid point (0,0). If no z-level variable is found, it checks for s-level
    variables ('s_rho', 's_w', 's_rho_u', 's_rho_v') and returns the first one found.

    Parameters:
        dataset (xarray.Dataset): Input dataset.

    Returns:
        tuple: A tuple containing:
            - z_levels (np.ndarray): The z-level or s-level array.
            - z_var_name (str): The name of the z-level or s-level variable found.
                                 Returns None, None if no z or s level variable is found.

    Raises:
        Warning: If no z-level variable is found, a warning is printed and s-levels are checked.
                 If no s-level variable is also found, another warning is printed and None, None is returned.
    """
    z_vars = ["z_rho", "z_w", "z_rho_u", "z_rho_v"]
    for z_var in z_vars:
        if z_var in dataset:
            return dataset[z_var][0, :, 0, 0].values, z_var

    print("Warning: No z-level variable found in the dataset.")
    s_vars = ["s_rho", "s_w", "s_rho_u", "s_rho_v"]
    for s_var in s_vars:
        if s_var in dataset:
            print("Returning s-levels instead.")
            return dataset[s_var].values, s_var
    print("Warning: No s-level variable found in the dataset.")
    print("Returning None for z-levels.")
    return None, None


def get_depth_index(z_levs, depth):
    """
    Finds the index in the z-levels array that corresponds to a given depth.

    This function searches for the index in the 'z_levs' array that is closest to the specified
    'depth'. It assumes 'z_levs' are negative values representing depth below the surface.

    Parameters:
        z_levs (np.ndarray): Array of depth levels (negative values, increasing downwards).
        depth (float): Target depth in meters (positive value).

    Returns:
        int: The index in 'z_levs' corresponding to the closest depth level. Returns 0 (bottom index)
             if the target depth exceeds the maximum depth in 'z_levs'.

    Raises:
        Warning: If the target depth exceeds the maximum depth, a warning is printed, and the bottom index is returned.
    """
    z = np.abs(z_levs)
    if depth >= z.max():
        print(f"Warning: Depth {depth} exceeds the maximum depth {z.max()}. Using bottom index.")
        return 0 # Bottom index
    return (z <= depth).argmax() - 1


"""
###########################
PLOTTING FUNCTIONS
###########################
"""
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


    for j, time in enumerate(time_idx):
        for i, dataset in enumerate(datasets):
            idx = j * n_datasets + i
            if eta is not None or xi is not None: #plot depth
                im = plot_depth(dataset, variable, time, ax=ax[idx], eta=eta, xi=xi, z_levs=z_levs, vmin=minmax[0], vmax=minmax[1], max_depth=max_depth, cbar=False, gen_levels=gen_levels)
            else: #plot surface
                _ , im = plot_surface(dataset, variable, time, ax=ax[idx], vmin=minmax[0], vmax=minmax[1], cbar=False, s_rho=s_rho)

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
    cbar = fig.colorbar(im, cax=cbar_ax)
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
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(f"{dataset[variable].attrs['long_name']} ({dataset[variable].attrs['units']})", fontsize=8)


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


    # Plot each time index
    for i in range(len(time_idx)):
        plot_surface(dataset, variable, time_idx[i], ax=axs[i], vmin=vmin, vmax=vmax, cbar=False)
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
    cbar = fig.colorbar(im, cax=cbar_ax)
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
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"{dataset[variable].attrs['long_name']} ({dataset[variable].attrs['units']})", fontsize=8)


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



def plot_over_time(dataset, data, data_name, title, ax=None):
    """
    Plots a time series of the given data.

    Parameters:
        dataset (xarray.Dataset): The dataset (used for time coordinate).
        data (np.ndarray): 1D array of data to plot over time.
        data_name (str): Label for the data (y-axis label).
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    time = dataset.time.values / (30 * 24 * 60 * 60)  # convert time from seconds to months

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, data)
    ax.set_title(title)
    ax.set_xlabel('Time (months)')
    ax.set_ylabel(data_name)
    plt.grid(True)


def plot_weighted_average(dataset, variable, title=None, ax=None):
    """
    Plots the time series of the volume-weighted average of a variable.

    Uses 'compute_weighted_average' to calculate the weighted average.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        variable (str): The name of the variable to plot the weighted average for.
        title (str, optional): Title of the plot. Defaults to None (auto-generated).
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    weighted_avg = compute_weighted_average(dataset, variable)
    var_data = dataset[variable]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    plot_over_time(dataset, weighted_avg, "{} ({})".format(var_data.long_name, var_data.units), title, ax)
    ax.set_ylabel("{} ({})".format(var_data.long_name, var_data.units))

def plot_surface_average(dataset, variable, title=None, ax=None):
    """
    Plots the time series of the surface-weighted average of a variable.

    Uses 'compute_surface_average' to calculate the surface average.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        variable (str): The name of the variable to plot the surface average for.
        title (str, optional): Title of the plot. Defaults to None (auto-generated).
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    avg = compute_surface_average(dataset, variable)
    var_data = dataset[variable]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    plot_over_time(dataset, avg, "{} ({})".format(var_data.long_name, var_data.units), title, ax)
    ax.set_ylabel("{} ({})".format(var_data.long_name, var_data.units))


def plot_total_KE(dataset, title="", ax=None):
    """
    Plots the time series of the total kinetic energy (KE) of the flow.

    Uses 'compute_total_KE' to calculate the total KE.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        title (str, optional): Title of the plot. Defaults to "".
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    total_KE = compute_total_KE(dataset)

    plot_over_time(dataset, total_KE, 'Kinetic Energy (J)', title, ax)



def plot_average_KE(dataset, title="", ax=None):
    """
    Plots the time series of the average kinetic energy (KE) of the flow.

    Uses 'compute_average_KE' to calculate the average KE.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        title (str, optional): Title of the plot. Defaults to "".
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """

    avg_KE = compute_average_KE(dataset)

    plot_over_time(dataset, avg_KE, 'Average Kinetic Energy (m^2/s^2)', title, ax)
