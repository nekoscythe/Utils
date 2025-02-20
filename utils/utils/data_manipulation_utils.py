import numpy as np
import xarray as xr

def interpolate_to_rho(var_data):
    """
    Efficiently interpolates a variable to the rho grid points using NumPy vectorized operations.

    Parameters:
        var_data (xarray.DataArray): The variable data to interpolate.

    Returns:
        np.ndarray: The interpolated data on the rho grid.
    """
    def avg_xi(x):
        return 0.5 * (x[..., :-1] + x[..., 1:])
    def avg_eta(x):
        return 0.5 * (x[..., :-1, :] + x[..., 1:, :])
    def avg_s_rho(x):
        return 0.5 * (x[..., :-1, :, :] + x[..., 1:, :, :])
    
    dims = var_data.dims
    if 'xi_u' in var_data.dims:
        axis = dims.index('xi_u')
        xi_u = var_data['xi_u'].values
        xi_rho = avg_xi(xi_u)
        var_data = xr.apply_ufunc(
            avg_xi,
            var_data,
            input_core_dims=[dims],
            output_core_dims=[dims],
            exclude_dims={'xi_u'},
            dask='allowed'
        ).rename({'xi_u': 'xi_rho'})
        var_data['xi_rho'] = xi_rho
        
    if 'eta_v' in var_data.dims:
        axis = dims.index('eta_v')
        eta_v = var_data['eta_v'].values
        eta_rho = avg_eta(eta_v)
        var_data = xr.apply_ufunc(
            avg_eta,
            var_data,
            input_core_dims=[dims],
            output_core_dims=[dims],
            exclude_dims={'eta_v'},
            dask='allowed'
        ).rename({'eta_v': 'eta_rho'})
        var_data['eta_rho'] = eta_rho
    if 's_w' in var_data.dims:
        axis = dims.index('s_w')
        s_w = var_data['s_w'].values
        s_rho = avg_s_rho(s_w)
        var_data = xr.apply_ufunc(
            avg_s_rho,
            var_data,
            input_core_dims=[dims],
            output_core_dims=[dims],
            exclude_dims={'s_w'},
            dask='allowed'
        ).rename({'s_w': 's_rho'})
        var_data['s_rho'] = s_rho
        
        
    return var_data
    
    
    
def lat_km_to_eta(dataset, lat_km):
    return int((lat_km - dataset.lat_km[0,0]) / (dataset.lat_km[-1,0] - dataset.lat_km[0,0]) * dataset.eta_rho.size)

def lon_km_to_xi(dataset, lon_km):
    return int((lon_km - dataset.lon_km[0,0]) / (dataset.lon_km[0,-1] - dataset.lon_km[0,0]) * dataset.xi_rho.size)


def slice_data(dataset, variable, time_idx, lat=None, lon=None):
    """
    Extracts a depth slice from the dataset along either the latitude (eta) or longitude (xi) axis.

    Parameters:
        dataset (xarray.Dataset): Input dataset.
        variable (str): Variable to extract.
        time_idx (int): Time index for slicing.
        lat (int, optional): Latitude index for slicing. Defaults to None.
        lon (int, optional): Longitude index for slicing. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - data (np.ndarray): Extracted depth profile (interpolated).
            - X (np.ndarray): Corresponding coordinate values.
            - xlabel (str): X-axis label ("Longitude (km)" or "Latitude (km)").
    
    Raises:
        ValueError: If both or neither of `lat` and `lon` are provided.
    """
    if (lat is None) == (lon is None):  # Ensure exactly one of lat/lon is given
        raise ValueError("Specify exactly one of `lat` or `lon`, but not both.")

    # Extract data at the given time index
    data = dataset[variable].isel(time=time_idx)

    # Slice along the selected dimension
    if lat is not None:
        # convert lat to eta
        eta = lat_km_to_eta(dataset, lat)
        data = data[:, eta]  # Slice along eta
        X = dataset.get("lon_km", dataset.lon_rho)[eta]
        xlabel = "Longitude (km)"
        slice_idx = eta
    else:
        # convert lon to xi
        xi = lon_km_to_xi(dataset, lon)
        data = data[..., :, xi]  # Slice along xi
        X = dataset.get("lat_km", dataset.lat_rho)[:, xi]
        xlabel = "Latitude (km)"
        slice_idx = xi
        
        
    # Interpolate to rho points after slicing
    data = interpolate_to_rho(data)

    return data, X, xlabel, slice_idx
