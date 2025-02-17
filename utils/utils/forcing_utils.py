import numpy as np
import xarray as xr

def convertToSin(data, field, amplitude):
    """
    Converts a specified field in the dataset to a sinusoidal profile along the Y-dimension.

    The function replaces the values of the given field with a sine wave that varies along
    the Y-dimension (eta). The amplitude and spatial scale of the sine wave are controlled
    by the 'amplitude' parameter and the domain size.

    Parameters:
        data (xarray.Dataset): The dataset to modify.
        field (str): The name of the field to convert to a sine wave.
        amplitude (float): The amplitude of the sine wave.

    Returns:
        xarray.Dataset: A copy of the dataset with the specified field modified.
    """
    data_copy = data.copy()
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
    """
    Sets a linear gradient for a specified field along the Y-dimension.

    The function modifies the values of the given field to create a linear gradient
    from 'south_value' to 'north_value' along the Y-dimension (eta).

    Parameters:
        data (xarray.Dataset): The dataset to modify.
        field (str): The name of the field to apply the linear gradient to.
        south_value (float): The value at the southern boundary (minimum Y).
        north_value (float): The value at the northern boundary (maximum Y).

    Returns:
        xarray.Dataset: A copy of the dataset with the specified field modified.
    """
    data_copy = data.copy()
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
