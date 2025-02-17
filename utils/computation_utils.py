import numpy as np
from .depth_utils import get_zlevs, get_depth_index # Import at the beginning
from .data_manipulation_utils import interpolate_to_rho # Import at the beginning
import xarray as xr

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
    data = interpolate_to_rho(data)
    dA = ds.dA.values
    if dA.ndim == 2: # Assuming it's (lat, lon)
        dA = dA.reshape((1, dA.shape[0], dA.shape[1]))
    elif dA.ndim != 3:
        raise ValueError("dA should have 2 or 3 dimensions (lat, lon) or (time, lat, lon)")
    weighted = data * dA
    return np.mean(weighted, axis=(1, 2)) / np.mean(dA)


def compute_total(dataset, variable, depth=-1):
    """
    Calculate the total of a given variable in the entire domain (or till specific depth) over time.
    
    Parameters:
        dataset (xarray.Dataset): The dataset containing the variable.
        variable (str): The name of the variable to calculate the total for.
        depth (int, optional): The depth in meters to calculate the total for.
                               Defaults to -1, which means total over all depths.
    """
    data = dataset[variable]
    data_vals = interpolate_to_rho(data)
    dV = dataset.dV.values

    if depth != -1:
        z_levs , _ = get_zlevs(dataset)
        idx = get_depth_index(z_levs, depth)
        data_vals = data_vals[:, :idx+1]
        dV = dV[:, :idx+1]

    total = np.sum(data_vals * dV, axis=(1, 2, 3))

    return total

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
    total = compute_total(dataset, variable, depth)
    dV = dataset.dV.values

    if depth != -1:
        z_levs , _ = get_zlevs(dataset)
        idx = get_depth_index(z_levs, depth)
        dV = dV[:, :idx+1]
    
    total_volume = np.sum(dV, axis=(1, 2, 3))
    weighted_avg_over_time = total / total_volume

    return weighted_avg_over_time