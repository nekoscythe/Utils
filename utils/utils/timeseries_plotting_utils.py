import matplotlib.pyplot as plt
from .helper_plotting_utils import convert_time

def plot_over_time(dataset, data, data_name, title, ax=None, time_unit="days", label=None):
    """
    Plots a time series of the given data.

    Parameters:
        dataset (xarray.Dataset): The dataset (used for time coordinate).
        data (np.ndarray): 1D array of data to plot over time.
        data_name (str): Label for the data (y-axis label).
        title (str): Title of the plot.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    time = dataset.time.values
    time, time_unit = convert_time(time, time_unit)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, data, label=label)
    ax.set_title(title)
    ax.set_xlabel(f'Time ({time_unit})')
    ax.set_ylabel(data_name)
    ax.grid(True)
    ax.legend()


def plot_weighted_average(dataset, variable, title=None, ax=None, time_unit='days', label=None):
    """
    Plots the time series of the volume-weighted average of a variable.

    Uses 'compute_weighted_average' to calculate the weighted average.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        variable (str): The name of the variable to plot the weighted average for.
        title (str, optional): Title of the plot. Defaults to None (auto-generated).
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    weighted_avg = compute_weighted_average(dataset, variable)
    var_data = dataset[variable]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    plot_over_time(dataset, weighted_avg, "{} ({})".format(var_data.long_name, var_data.units), title, ax, time_unit,label=label)
    ax.set_ylabel("{} ({})".format(var_data.long_name, var_data.units))

def plot_surface_average(dataset, variable, title=None, ax=None, time_unit='days', label=None):
    """
    Plots the time series of the surface-weighted average of a variable.

    Uses 'compute_surface_average' to calculate the surface average.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        variable (str): The name of the variable to plot the surface average for.
        title (str, optional): Title of the plot. Defaults to None (auto-generated).
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    avg = compute_surface_average(dataset, variable)
    var_data = dataset[variable]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    plot_over_time(dataset, avg, "{} ({})".format(var_data.long_name, var_data.units), title, ax, time_unit,label=label)
    ax.set_ylabel("{} ({})".format(var_data.long_name, var_data.units))


def plot_total_KE(dataset, title="", ax=None, time_unit='days', label=None):
    """
    Plots the time series of the total kinetic energy (KE) of the flow.

    Uses 'compute_total_KE' to calculate the total KE.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        title (str, optional): Title of the plot. Defaults to "".
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    total_KE = compute_total(dataset, 'KE')

    plot_over_time(dataset, total_KE, 'Kinetic Energy (m^2/s^2)', title, ax, time_unit, label)


def plot_average_KE(dataset, title="", ax=None, time_unit='days', label=None):
    """
    Plots the time series of the average kinetic energy (KE) of the flow.

    Uses 'compute_average_KE' to calculate the average KE.

    Parameters:
        dataset (xarray.Dataset): The dataset to plot from.
        title (str, optional): Title of the plot. Defaults to "".
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created. Defaults to None.
    """
    
    avg_KE = compute_weighted_average(dataset, 'KE')

    plot_over_time(dataset, avg_KE, 'Average Kinetic Energy (m^2/s^2)', title, ax, time_unit, label)

from .computation_utils import compute_weighted_average, compute_surface_average, compute_total # Import here to avoid circular dependency
