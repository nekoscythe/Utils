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
    data = interpolate_to_rho(ds[var][:, s_rho])
    dA = ds.dA
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
    data = interpolate_to_rho(dataset[variable])
    dV = dataset.dV

    if depth != -1:
        z_levs , _ = get_zlevs(dataset)
        idx = get_depth_index(z_levs, depth)
        data = data[:, :idx+1]
        dV = dV[:, :idx+1]

    # Compute the total by summing over depth, lat, and lon
    total = (data * dV).sum(dim=("s_rho", "eta_rho", "xi_rho"))

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
    dV = dataset.dV

    if depth != -1:
        z_levs , _ = utils.get_zlevs(dataset)
        idx = utils.get_depth_index(z_levs, depth)
        dV = dV[:, :idx+1]
    
    total_volume = np.sum(dV, axis=(1, 2, 3))
    weighted_avg_over_time = total / total_volume

    return weighted_avg_over_time