import numpy as np
import xarray as xr
import os  # Import the os module for path manipulation
import xroms # Import xroms library
from .geo_utils import add_km_grid
from .general_utils import reset_time, is_data_positive  # Moved reset_time and added is_data_positive import

def get_savetime_hours(dataset):
    """
    Calculates the time step of the dataset in hours.

    Parameters:
        dataset (xarray.Dataset): The input dataset containing a 'time' coordinate.

    Returns:
        float: The time step in hours.
    """
    if len(dataset.time) < 2: # Input validation: check for at least two time points
        raise ValueError("Dataset must have at least two time points to calculate time step.")
    time = dataset.time
    diff = time[1] - time[0]
    return diff.values / 3600


def add_KE(dataset):
    """
    Compute the kinetic energy (KE) of the flow and return it as a DataArray.

    KE is calculated as 0.5 * (u_mid^2 + v_mid^2), where u_mid and v_mid are the
    horizontal velocity components interpolated to the rho points.
    Does not modify the input dataset.

    Parameters:
        dataset (xarray.Dataset): The dataset containing 'u' and 'v' velocity variables.

    Returns:
        xarray.DataArray: Kinetic energy DataArray.
    """
    if 'u' not in dataset or 'v' not in dataset: # Input validation: check for u and v
        raise ValueError("Dataset must contain variables 'u' and 'v' to compute KE.")

    dataset_copy = dataset.copy(deep=True) # Work on a copy to avoid in-place modification
    u = dataset_copy.u.values
    v = dataset_copy.v.values

    u_mid = 0.5 * (u[..., :-1] + u[..., 1:])  # Interpolate u to rho points
    v_mid = 0.5 * (v[..., :-1, :] + v[..., 1:, :])  # Interpolate v to rho points

    KE = 0.5 * (u_mid**2 + v_mid**2) #this is just 1/2 * velocity^2
    
    dataset_copy['KE'] = (('time', 's_rho', 'eta_rho', 'xi_rho'), KE, {'long_name' : 'kinetic energy', 'units': 'meter2 second-2'}) #add KE to dataset

    dataset_copy['KE'] = dataset_copy.KE.chunk() # rechunk to match original dataset
    KE_da = dataset_copy['KE']

    return KE_da


def add_RV(dataset):
    """
    Compute the relative vorticity (RV) of the flow and return it as a DataArray.

    RV is calculated as dv/dx - du/dy, where u and v are horizontal velocity components
    and derivatives are approximated using finite differences on the grid.
    Does not modify the input dataset.

    Assumes a regular grid for finite difference calculation.

    Parameters:
        dataset (xarray.Dataset): The dataset containing 'u', 'v', 'pm', and 'pn' variables.

    Returns:
        xarray.DataArray: Relative vorticity DataArray.
    """
    required_vars = ['u', 'v', 'pm', 'pn'] # Input validation: check for required variables
    for var in required_vars:
        if var not in dataset:
            raise ValueError(f"Dataset must contain variable '{var}' to compute RV.")
        
    dataset_copy = dataset.copy(deep=True) # Work on a copy to avoid in-place modification

    pm = dataset_copy.pm.values #1/dx
    pn = dataset_copy.pn.values #1/dy

    pm_psi = 0.25 * (pm[:-1, :-1] + pm[1:, :-1] + pm[:-1, 1:] + pm[1:, 1:]) # interpolate pm to psi points
    pn_psi = 0.25 * (pn[:-1, :-1] + pn[1:, :-1] + pn[:-1, 1:] + pn[1:, 1:]) # interpolate pn to psi points

    dv = dataset_copy.v[:,:,:, 1:].values - dataset_copy.v[:,:,:, :-1].values
    du = dataset_copy.u[:,:, 1:, :].values - dataset_copy.u[:,:, :-1, :].values


    zeta_vort = (dv * pm_psi) - (du * pn_psi) # dv/dx - du/dy
    
    dataset_copy['RV'] =  (('time', 's_rho', 'eta_v', 'xi_u'), zeta_vort, {'long_name' : 'relative vorticity', 'units': 'second-1'})
    RV_da = dataset_copy['RV']

    
    return RV_da


def add_dV(dataset):
    """
    Compute the volume (dV) of each cell in the grid and return it as a DataArray.

    dV is calculated as dA * dz, where dA is the surface area of the cell and dz is the
    vertical thickness.
    Does not modify the input dataset.
    Assumes z_w is structured for np.diff(z_w, axis=1) to calculate dz.

    Parameters:
        dataset (xarray.Dataset): The dataset containing 'dA' and 'z_w' variables.

    Returns:
        xarray.DataArray: Volume DataArray.
    """
    required_vars = ['dA', 'z_w'] # Input validation: check for required variables
    for var in required_vars:
        if var not in dataset:
            raise ValueError(f"Dataset must contain variable '{var}' to compute dV.")

    dataset_copy = dataset.copy(deep=True) # Work on a copy to avoid in-place modification
    dz = np.diff(dataset.z_w, axis=1)

    dV = dataset.dA.values * dz  # volume of each cell
    
    dataset_copy['dV'] = (('time', 's_rho', 'eta_rho', 'xi_rho'), dV, {'long_name' : 'volume of cells on RHO grid' , 'units': 'meter3'})

    dataset_copy['dV'] = dataset_copy.dV.chunk() # Chunk the DataArray for better performance
    dV_da = dataset_copy.dV
    return dV_da

def fix_dV(dataset):
    """
    """
    #if the dims of dV are eta and xi, rename them to eta_rho and xi_rho
    if 'eta' in dataset.dV.dims and 'xi' in dataset.dV.dims:
        dataset['dV'] = dataset['dV'].rename({'eta': 'eta_rho', 'xi': 'xi_rho'})
    return dataset


def preprocess(dataset, preprocess_done=False, output_path=None):
    """
    Applies a series of preprocessing steps to the input dataset.
    Does not modify the input dataset in-place.
    Optionally saves the *additional* preprocessed variables to a new NetCDF file.

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
        output_path (str, optional): Path to save the *additional* preprocessed variables to a new NetCDF file.
                                     If None, the preprocessed variables are not saved to a file. Defaults to None.

    Returns:
        xarray.Dataset: The preprocessed dataset (including original and preprocessed variables).
                        The original dataset is not modified.
    """
    processed_dataset = dataset.copy() # work on a copy to avoid in-place modification
    if not preprocess_done: 
        rv_da = add_RV(processed_dataset) # calculate derived variables as DataArrays
        
    processed_dataset = remove_ghost_points(processed_dataset) # remove ghost points from the copy
    
    if not preprocess_done:
        ke_da = add_KE(processed_dataset)
        dv_da = add_dV(processed_dataset)

        processed_dataset = processed_dataset.assign({'RV': rv_da, 'KE': ke_da, 'dV': dv_da}) # assign derived variables to processed_dataset
        add_km_grid(processed_dataset) # add km_grid to processed_dataset

    processed_dataset = non_negative_bio(processed_dataset)
    processed_dataset = reset_time(processed_dataset) # Use imported reset_time from general_utils

    if output_path and not preprocess_done: # Save *additional* variables to NetCDF if output_path is provided
        additional_vars_ds = processed_dataset[['RV', 'KE', 'dV', 'lon_km', 'lat_km']] # create dataset with only additional vars
        additional_vars_ds.to_netcdf(output_path)
        print(f"Additional preprocessed variables saved to: {output_path}")
        
    return processed_dataset

def fast_merge(dataset, additional_vars_ds):
    """
    Merges the original dataset with the additional variables dataset.
    Does not modify the input datasets in-place, returns a new dataset.

    Parameters:
        dataset (xarray.Dataset): The original dataset.
        additional_vars_ds (xarray.Dataset): The dataset containing additional variables to merge.

    Returns:
        xarray.Dataset: The merged dataset.
    """
    data_vars = additional_vars_ds.data_vars # get data variables from additional_vars_ds
    for data_var in data_vars: # loop over data variables
        dataset[data_var] = additional_vars_ds[data_var] # add data variables to original dataset
    #rechunking
    one_var_size = dataset.salt.nbytes / 1e9 # in GB
    # we want to have 1 GB chunks
    chunks = int(dataset.time.size/one_var_size)
    dataset = dataset.chunk({'time': chunks})
    return dataset
    

def load_and_preprocess(file_path):
    """
    Loads a NetCDF dataset and applies preprocessing.
    Checks for a preprocessed file and loads from it if available to avoid redundant computations.

    Parameters:
        file_path (str): Path to the original NetCDF dataset file.
        bio (bool, optional): If True, apply non-negativity to biological variables during preprocessing. Defaults to False.

    Returns:
        xarray.Dataset: The preprocessed dataset.
    """
    preprocessed_file_path = file_path.replace(".nc", "_preprocessed.nc") # Define path for preprocessed file

    if os.path.exists(preprocessed_file_path): # Check if preprocessed file exists
        print(f"Loading preprocessed variables from: {preprocessed_file_path}")
        original_dataset = xroms.open_netcdf(file_path) # load original dataset
        print("Original dataset loaded")
        #preprocces to match the additional variables
        original_dataset = preprocess(original_dataset, preprocess_done=True)
        print("Preprocessing done. Loading additional variables")
        additional_vars_ds = xr.open_dataset(preprocessed_file_path) # Load preprocessed variables
        print("Additional variables loaded. Merging datasets...")
        preprocessed_dataset = fast_merge(original_dataset, additional_vars_ds) # Merge datasets
        print("Additional variables loaded")
        preprocessed_dataset = fix_dV(preprocessed_dataset)
    else: # Preprocessed file does not exist, perform preprocessing and save
        print(f"Preprocessing dataset from: {file_path}")
        dataset = xroms.open_netcdf(file_path) # Load original dataset
        print("Original dataset loaded. Preprocessing...")
        preprocessed_dataset = preprocess(datasetb, output_path=preprocessed_file_path) # Preprocess and save additional vars
        print("Preprocessing done.")
        preprocessed_dataset = fix_dV(preprocessed_dataset)
    return preprocessed_dataset


def remove_ghost_points(dataset):
    """
    Removes ghost points from the eta_rho and xi_rho dimensions of the dataset.
    Does not modify the input dataset in-place, returns a new dataset.

    Assumes ghost points are the first and last indices of 'eta_rho' and 'xi_rho' dimensions.

    Parameters:
        dataset (xarray.Dataset): The dataset to remove ghost points from.

    Returns:
        xarray.Dataset: The dataset with ghost points removed.
    """
    dims_to_check = ['eta_rho', 'xi_rho'] # Input validation: check for required dimensions
    for dim in dims_to_check:
        if dim not in dataset.dims:
            raise ValueError(f"Dataset must have dimension '{dim}' to remove ghost points.")
    return dataset.isel(eta_rho=slice(1,-1), xi_rho=slice(1,-1))


def non_negative_bio(dataset, bio_vars = ['CHLA', 'NO3', 'PHYTO']):
    """
    Clips biological variables to ensure non-negative values.
    Does not modify the input dataset in-place, returns a new dataset.

    Parameters:
        dataset (xarray.Dataset): The dataset to modify.
        bio_vars (list, optional): List of biological variables to clip.
                                     Defaults to ['CHLA', 'NO3', 'PHYTO'].

    Returns:
        xarray.Dataset: The dataset with non-negative biological variables.
    """
    dataset_copy = dataset.copy() # avoid in-place modification
    for var in bio_vars: # Flexibility: loop over bio_vars list
        if var in dataset_copy: # Robustness: check if var exists before clipping
            dataset_copy[var] = dataset_copy[var].clip(min=0)
        else:
            print(f"Warning: Biological variable '{var}' not found in dataset, skipping non-negativity clipping for this variable.")
    return dataset_copy
