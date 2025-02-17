import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


##################
# Plotting Helpers
##################

def convert_time(time, unit="days"):
    """
    Converts time values to a specified unit.

    This function converts time values to a specified unit (e.g., months, days, years).

    Parameters:
        time (xarray.DataArray): The time values to convert (in seconds).s
        unit (str, optional): The unit to convert to. Defaults to "days".

    Returns:
        xarray.DataArray: The converted time values.
    """
    # convert time to np.timedelta64 in seconds
    time = np.asarray(time, dtype='timedelta64[s]')
    if unit == "months":
        return time / np.timedelta64(30, 'D'), unit
    elif unit == "minutes":
        return time / np.timedelta64(1, 'm'), unit
    elif unit == "seconds":
        return time / np.timedelta64(1, 's'), unit
    elif unit == "hours":
        return time / np.timedelta64(1, 'h'), unit
    elif unit == "days":
        return time / np.timedelta64(1, 'D'), unit
    elif unit == "weeks":
        return time / np.timedelta64(7, 'D'), unit
    elif unit == "years":
        return time / np.timedelta64(360, 'D'), unit
    else:
        #default to days
        return time / np.timedelta64(1, 'D'), "days"



def get_optimal_subplot_dims(n, max_columns=4):
    """
    Determine the optimal subplot dimensions (rows, columns) for a given number of plots.

    The function aims to arrange subplots in a grid that is as close to square as possible,
    with a maximum number of columns specified by 'max_columns'.

    Parameters:
        n (int): The number of subplots to arrange.
        max_columns (int, optional): The maximum number of columns in the subplot grid. Defaults to 4.

    Returns:
        tuple: A tuple (rows, columns) representing the optimal subplot dimensions.
    """
    # Function to find factors of a number
    factors = []
    for i in range(1, int(np.sqrt(n)) + 1):
        if n % i == 0:
            factors.append((i, n // i))

    # Find the optimal factor pair (rows, columns)
    best_rows, best_cols = None, None
    for rows, cols in factors:
        if cols <= max_columns:
            if best_rows is None or abs(rows - cols) < abs(best_rows - best_cols):
                best_rows, best_cols = rows, cols

    # If no optimal pair found, we need to ensure columns don't exceed the max
    if best_rows is None or best_cols > max_columns:
        # Use default behavior if no suitable pair found (e.g., columns = max_columns)
        best_cols = min(n, max_columns)
        best_rows = (n + best_cols - 1) // best_cols

    return best_rows, best_cols

def create_subplots(n, max_columns=4, figsize=None):
    """
    Creates a figure and a set of subplots arranged in an optimal grid.

    Uses 'get_optimal_subplot_dims' to determine the subplot layout.

    Parameters:
        n (int): The number of subplots to create.
        max_columns (int, optional): The maximum number of columns in the subplot grid. Defaults to 4.
        figsize (tuple, optional): The figure size (width, height) in inches. Defaults to None,
                                    which uses a default size based on the number of columns and rows.

    Returns:
        tuple: A tuple (fig, axes) where 'fig' is the matplotlib Figure object and 'axes'
               is a list of matplotlib Axes objects (flattened if more than one subplot).
    """
    rows, cols = get_optimal_subplot_dims(n, max_columns)

    if figsize is not None:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4)) # 4x4 aspect ratio

    # Flatten the axes if necessary
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    # Remove any extra axes if there are fewer than n subplots
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    return fig, axes

def get_colormap_and_limits(data, vmin=None, vmax=None):
    """
    Determine colormap and value limits for plotting based on the data range.

    If the data is entirely positive, 'turbo' colormap is used. Otherwise, 'seismic'
    colormap is used to represent both positive and negative values.

    Parameters:
        data (np.ndarray): The data to be plotted.
        vmin (float, optional): Minimum value for the colormap. If None, it's determined from data. Defaults to None.
        vmax (float, optional): Maximum value for the colormap. If None, it's determined from data. Defaults to None.

    Returns:
        tuple: A tuple (cmap, vmin, vmax) containing the colormap name, minimum value, and maximum value.
    """
    pos_data = (data >= 0).all()  # Improved readability
    cmap = "turbo" if pos_data else "seismic"
    vmax = np.nanmax(np.abs(data)) if vmax is None else vmax
    vmin = np.nanmin(data) if vmin is None else vmin
    return cmap, vmin, vmax



def make_cbar(var_data, ax, im):
    """
    Adds a colorbar to a plot.
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    cbar = ax.get_figure().colorbar(im, cax=cax)
    cbar.set_label(f"{var_data.attrs['units']}")
    return cbar


def add_colorbar(fig, im, var_data, ax=None):
    """
    Adds a colorbar to a figure.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to add the colorbar to.
        im (matplotlib.collections.QuadMesh): The image object returned by pcolormesh or similar.
        var_data (xarray.DataArray): The data variable being plotted (used for units).
        ax (matplotlib.axes.Axes, optional):  Axes object to anchor the colorbar to, if None, it uses the last axes. Defaults to None.
    """
    if ax is None:
        cbar = fig.colorbar(im)
    else: #if ax is provided, use make_axes_locatable for more controlled placement next to the ax
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"{var_data.attrs['units']}")
    return cbar


def set_plot_labels(ax, var_data, xlabel=None, ylabel=None, title=None, time_value=None, time_unit='months'):
    """
    Sets common plot labels and titles.

    Parameters:
        ax (matplotlib.axes.Axes):         The axes object to set labels for.
        var_data (xarray.DataArray):      The data variable being plotted (used for long_name and units in title).
        xlabel (str, optional):           Label for the x-axis. Defaults to None.
        title (str, optional):            Title for the plot. Defaults to None (auto-generated from var_data).
        time_value (float, optional):      Time value to be displayed in the title. Defaults to None (no time in title).
        time_unit (str, optional):         Unit of time for the title, e.g., 'seconds', 'days', 'months', 'years'. Defaults to 'months'.
    """
    if title is None:
        title = f"{var_data.attrs['long_name']}" # Default title without time
        if time_value is not None: # Only add time to title if time_value is provided
            time_value, time_unit = convert_time(time_value, unit=time_unit) # convert time to specified unit
            title = f"{var_data.attrs['long_name']} at {time_value:.1f} {time_unit}" # add time to title
        else:
            title = f"{var_data.attrs['long_name']}"
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    
