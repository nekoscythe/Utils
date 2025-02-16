import numpy as np
from .depth_utils import get_zlevs, get_depth_index # Import at the beginning

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
    if dA.ndim == 2: # Assuming it's (lat, lon)
        dA = dA.reshape((1, dA.shape[0], dA.shape[1]))
    elif dA.ndim != 3:
        raise ValueError("dA should have 2 or 3 dimensions (lat, lon) or (time, lat, lon)")
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
