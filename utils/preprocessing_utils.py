import numpy as np
from .geo_utils import add_km_grid
from .general_utils import reset_time

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
