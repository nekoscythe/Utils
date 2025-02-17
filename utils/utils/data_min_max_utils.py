import numpy as np
from .depth_utils import get_depth_index, get_zlevs # Import here to avoid circular dependency


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
