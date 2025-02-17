import xarray as xr
import numpy as np
import matplotlib.pyplot as plt # Import matplotlib.pyplot


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


def is_data_positive(data):
    """
    Checks if all values in a numpy array are positive or zero.

    Parameters:
        data (np.ndarray): The input data array.

    Returns:
        bool: True if all values are positive or zero, False otherwise.
    """
    return np.all(data >= 0)

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



