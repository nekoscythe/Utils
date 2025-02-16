import numpy as np

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
