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
    time = dataset.time
    diff = time[1] - time[0]
    return diff.values / 3600

def add_KE(dataset):
    """
    Compute the square velocity of the flow at each grid cell and add it to the dataset inplace.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    """
    u_mid = 0.5 * (dataset.u[..., :-1].values + dataset.u[..., 1:].values) # interpolate u to rho points
    v_mid = 0.5 * (dataset.v[...,:-1,:].values + dataset.v[..., 1:,:].values) # interpolate v to rho points
    
    KE = 0.5 * (u_mid**2 + v_mid**2) #this is just 1/2 * velocity^2
    #add KE to dataset
    dataset['KE'] = (('time', 's_rho', 'eta_rho', 'xi_rho'), KE, {'long_name' : 'kinetic energy', 'units': 'meter2 second-2'})
    dataset['KE'] = dataset.KE.chunk()
    
def add_RV(dataset):
    """
    Compute the relative vorticity of the flow at each grid cell and add it to the dataset inplace.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
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
    Compute the volume of each cell in the grid and add them to the dataset.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    """
    dz = np.diff(dataset.z_w, axis=1)

    # dA = dx_expanded * dy_expanded  # surface area of each cell
    dV = dataset.dA.values * dz  # volume of each cell
    
    dataset['dV'] = (('time', 's_rho', 'eta', 'xi'), dV, {'long_name' : 'volume of cells on RHO grid' , 'units': 'meter3'})

def reset_time(dataset):
    dataset['time'] = dataset.time - dataset.time[0]
    return dataset
    
def preprocess(dataset, bio=False):
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
    data = ds[var][:, s_rho]
    dA = ds.dA.values
    dA = dA.reshape((1, dA.shape[0], dA.shape[1]))
    weighted = data * dA
    return np.mean(weighted, axis=(1, 2)) / np.mean(dA)


def compute_weighted_average(dataset, variable, depth=-1):
    """
    Calculate the weighted average of a given variable over time.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    variable (str): The name of the variable to calculate the weighted average for.
    depth (int, optional): The depth in meters to calculate the weighted average for. Defaults to -1.
    
    Returns:
    np.ndarray: The weighted average of the variable over time.
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
    Compute the total kinetic energy of the flow over time.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    
    Returns:
    np.ndarray: The total kinetic energy of the flow over time.
    """
    
    dV = dataset.dV.values # volume of cell centered at rho points
    densities = dataset.rho.values + dataset.rho0.values # remove ghost cells and add rho0
    total_KE = dataset.KE.values * dV * densities # multiply by volume and density to get kinetic energy
    
    total_KE_over_time = np.sum(total_KE, axis=(1, 2, 3))
    
    return total_KE_over_time


def compute_average_KE(dataset):
    """
    Compute the average square velocity of the flow over time.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    
    Returns:
    np.ndarray: The average square velocity of the flow over time.
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
    Determine colormap and value limits for the plot.
    
    Parameters:
        data (np.ndarray): Data to plot.
        vmin (float, optional): Minimum value for colormap. Defaults to None.
        vmax (float, optional): Maximum value for colormap. Defaults to None.
    
    Returns:
        tuple: Colormap name, minimum value, maximum value.
    """
    pos_data = (data >= 0).all()  # Improved readability
    cmap = "turbo" if pos_data else "seismic"
    vmax = np.nanmax(np.abs(data)) if vmax is None else vmax
    vmin = np.nanmin(data) if vmin is None else vmin
    return cmap, vmin, vmax



def make_cbar(var_data, ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)

    cbar = ax.get_figure().colorbar(im, cax=cax)
    cbar.set_label(f"{var_data.attrs['units']}")
    return cbar



######################
# Data Manipulation
######################

def scale_depth(z_levs, z_crit, max_multiplier=1, poly_order=1):
    z_crit = -1 * z_crit
    z_crit = max(z_levs.min(), z_crit)  # Ensure z_crit is within the range of z_levs

    # Compute scaling factor based on a power curve
    above_crit = (z_levs < z_crit) * max_multiplier
    below_crit = (z_levs >= z_crit) * (1 - (z_crit - z_levs) / z_crit)**poly_order * max_multiplier  # Cubic for sharper acceleration

    return below_crit + above_crit



def get_min_max(dataset, field, time, s_rho):
    min_val = 100000
    max_val = -100000
    for ds in dataset:
        min_val = min(min_val, np.min(ds[field].values[time, s_rho]))
        max_val = max(max_val, np.max(ds[field].values[time, s_rho]))
    return min_val, max_val

def print_minmax_biovars(ds):
    min_phyto, max_phyto = ds.PHYTO.min().values, ds.PHYTO.max().values
    min_no3, max_no3 = ds.NO3.min().values, ds.NO3.max().values
    min_chla, max_chla = ds.CHLA.min().values, ds.CHLA.max().values
    
    print(f"Min Phyto: {min_phyto:.2f}, Max Phyto: {max_phyto:.2f}, "
          f"Min NO3: {min_no3:.2f}, Max NO3: {max_no3:.2f}, "
          f"Min CHLA: {min_chla:.2f}, Max CHLA: {max_chla:.2f}")
    
def get_minmax_datasets(datasets, field, time=None, s_rho=None, eta=None, xi=None, max_depth=500, z_levs=None):
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
    


    

def get_max_biovars(datasets, time=-1, s_rho=-1):
    max_phyto = -1
    max_no3 = -1
    max_chla = -1
    min_phyto = 100000
    min_no3 = 100000
    min_chla = 100000
    
    for ds in datasets:
        if np.max(ds.PHYTO.values[time, s_rho]) > max_phyto:
            max_phyto = np.max(ds.PHYTO.values[time, s_rho])
        if np.max(ds.NO3.values[time, s_rho]) > max_no3:
            max_no3 = np.max(ds.NO3.values[time, s_rho])
        if np.max(ds.CHLA.values[time, s_rho]) > max_chla:
            max_chla = np.max(ds.CHLA.values[time, s_rho])
        if np.min(ds.PHYTO.values[time, s_rho]) < min_phyto:
            min_phyto = np.min(ds.PHYTO.values[time, s_rho])
        if np.min(ds.NO3.values[time, s_rho]) < min_no3:
            min_no3 = np.min(ds.NO3.values[time, s_rho])
        if np.min(ds.CHLA.values[time, s_rho]) < min_chla:
            min_chla = np.min(ds.CHLA.values[time, s_rho])
    
    return max_phyto, max_no3, max_chla, min_phyto, min_no3, min_chla
            
def plot_all(ds):
    for time in np.arange(0, len(ds.time), 1):
       plot_surface_biovars(ds, time)
       plot_depth_biovars(ds, time)
       plt.show()



def non_negative_bio(ds):
    ds['CHLA'] = ds.CHLA.clip(min=0)
    ds['NO3'] = ds.NO3.clip(min=0)
    ds['PHYTO'] = ds.PHYTO.clip(min=0)
    


def combine_along_time(datasets):
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
    # Define geodetic calculator (WGS84 ellipsoid)
    geod = Geod(ellps="WGS84")
    
    # Calculate the distance along latitude (north-south)
    x_km = np.zeros(len(longitudes))
    for i in range(1, len(longitudes)):
        _, _, dist = geod.inv(longitudes[i-1], latitude, longitudes[i], latitude)
        x_km[i] = x_km[i-1] + dist / 1000
    
    return x_km

def lat_to_km(latitudes, longitude):
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
    Convert arrays of latitudes and longitudes to distances in kilometers.

    Parameters:
    latitudes (array-like): Array of latitude values.
    longitudes (array-like): Array of longitude values.

    Returns:
    X_km (2D numpy array): X-coordinates in kilometers.
    Y_km (2D numpy array): Y-coordinates in kilometers.
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
    """Interpolate a variable to the rho points."""
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
    return dataset.isel(eta_rho=slice(1,-1), xi_rho=slice(1,-1))

def is_data_positive(data):
    """Check if the data is positive."""
    return np.all(data >= 0)



def slice_data(dataset, variable, time_idx, eta=None, xi=None):
    """
    Extract a slice from the dataset along either the eta or xi axis.
    
    Parameters:
        dataset (xarray.Dataset): Input dataset.
        variable (str): Variable to extract.
        time_idx (int): Time index for slicing.
        X (int, optional): Index along longitudinal dimension. Defaults to None.
        Y (int, optional): Index along latitudinal dimension. Defaults to None.
    
    Returns:
        tuple: Extracted data slice, corresponding coordinates, and eta-axis label.
    
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
    Identify z-level variable in the dataset.
    
    Parameters:
        dataset (xarray.Dataset): Input dataset.
    
    Returns:
        tuple: z-level array and the corresponding variable name.
    
    Raises:
        ValueError: If no z-level variable is found.
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
    Get the index corresponding to a given depth.
    
    Parameters:
        z_levs (xarray.DataArray): Depth levels.
        depth (float): Target depth in meters.
    
    Returns:
        int: Index of the closest depth.
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
    Plot the surface of a given variable at a specific time index.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    variable (str or numpy.ndarray): The variable to plot or its name as a string.
    time_idx (int or list): The time index (or indices) to plot.
    axs (matplotlib.axes.Axes, optional): Axes object for plotting.
    title (str, optional): Title for the plot.
    use_vmax (bool, optional): Whether to use the maximum value for color scaling.
    pos_data (bool, optional): Whether to use only positive data.
    shading (str, optional): Shading mode for pcolormesh.
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
    data: 2D numpy array of data at different depths
    z_levels: list of depths at which the data was collected
    max_depth: maximum depth to plot
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
    Plot data along depth for a given variable in the dataset.
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
    Plot data along depth for multiple time indices.
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
        
    vars = ["u", "v", "w", "temp", "salt", "rho", "KE", "RV", "zeta"]
    fig, ax = create_subplots(len(vars))
    
    for i, var in enumerate(vars):
        plot_surface(dataset, var, time, ax=ax.flat[i])
        ax[i].label_outer()
   
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    
    return fig, ax
    

def plot_depth_vars(dataset, time, eta=None, xi=None, max_depth=500, title=None):
    
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
    vars = ["PHYTO", "NO3"]
    fig, ax = create_subplots(len(vars), max_columns=max_columns, figsize=figsize)
    
    
    for i, var in enumerate(vars):
        plot_depth(dataset, var, time, max_depth=depth, ax=ax[i], eta=eta, xi=xi, shading=shading)
        ax[i].label_outer()
        
    # plt.tight_layout()
    
    return fig, ax
    
    


def plot_over_time(dataset, data, data_name, title, ax=None):
    """
    Plot the given data over time.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    data (np.ndarray): The data to plot over time.
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
    Plot the weighted average of a given variable over time.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    variable (str): The name of the variable to calculate the weighted average for.
    title (str): The title of the plot.
    """
    weighted_avg = compute_weighted_average(dataset, variable)
    var_data = dataset[variable]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    plot_over_time(dataset, weighted_avg, "{} ({})".format(var_data.long_name, var_data.units), title, ax)
    ax.set_ylabel("{} ({})".format(var_data.long_name, var_data.units))
    
def plot_surface_average(dataset, variable, title=None, ax=None):
    """
    Plot the average of a given variable over time.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    variable (str): The name of the variable to calculate the average for.
    title (str): The title of the plot.
    """
    avg = compute_surface_average(dataset, variable)
    var_data = dataset[variable]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    plot_over_time(dataset, avg, "{} ({})".format(var_data.long_name, var_data.units), title, ax)
    ax.set_ylabel("{} ({})".format(var_data.long_name, var_data.units))

    


def plot_total_KE(dataset, title="", ax=None):
    """
    Plot the total kinetic energy of the flow over time.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    title (str): The title of the plot.
    """
    total_KE = compute_total_KE(dataset)
    
    plot_over_time(dataset, total_KE, 'Kinetic Energy (J)', title, ax)
    
    
    
def plot_average_KE(dataset, title="", ax=None):
    """
    Plot the average kinetic energy of the flow over time.
    
    Parameters:
    dataset (xarray.Dataset): The dataset containing the variables.
    title (str): The title of the plot.
    """
    
    avg_KE = compute_average_KE(dataset)
    
    plot_over_time(dataset, avg_KE, 'Average Kinetic Energy (m^2/s^2)', title, ax)
    
    
    