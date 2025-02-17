import numpy as np
from pyproj import Geod

def lon_to_km(latitude, longitudes):
    """
    Converts an array of longitudes to kilometers, assuming a constant latitude.

    Calculates distances eastwards from the first longitude in the array.

    Parameters:
        latitude (float): The latitude at which to calculate distances.
        longitudes (np.ndarray): Array of longitude values.

    Returns:
        np.ndarray: Array of distances in kilometers corresponding to the longitudes.
    """
    # Define geodetic calculator (WGS84 ellipsoid)
    geod = Geod(ellps="WGS84")

    # Calculate the distance along latitude (north-south)
    x_km = np.zeros(len(longitudes))
    for i in range(1, len(longitudes)):
        _, _, dist = geod.inv(longitudes[i-1], latitude, longitudes[i], latitude)
        x_km[i] = x_km[i-1] + dist / 1000

    return x_km

def lat_to_km(latitudes, longitude):
    """
    Converts an array of latitudes to kilometers, assuming a constant longitude.

    Calculates distances northwards from the first latitude in the array.

    Parameters:
        latitudes (np.ndarray): Array of latitude values.
        longitude (float): The longitude at which to calculate distances.

    Returns:
        np.ndarray: Array of distances in kilometers corresponding to the latitudes.
    """
    # Define geodetic calculator (WGS84 ellipsoid)
    geod = Geod(ellps="WGS84")

    # Calculate the distance along longitude (east-west)
    y_km = np.zeros(len(latitudes))
    for j in range(1, len(latitudes)):
        _, _, dist = geod.inv(longitude, latitudes[j-1], longitude, latitudes[j])
        y_km[j] = y_km[j-1] + dist / 1000

    return y_km

def add_km_grid(dataset):
    """
    Adds kilometer-based longitude and latitude coordinates to the dataset.

    Creates 'lon_km' and 'lat_km' coordinates based on 'lon_rho' and 'lat_rho',
    representing distances in kilometers from the western and southern boundaries, respectively.

    Parameters:
        dataset (xarray.Dataset): The dataset to which kilometer grids should be added.
    """
    latitudes = dataset.lat_rho[:, 0].values
    longitudes = dataset.lon_rho[0, :].values

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError("latitudes and longitudes must be 1D arrays.")

    # Calculate the distance along latitude (north-south)
    y_km = lat_to_km(latitudes, longitudes[0])

    # Calculate the distance along longitude (east-west)
    x_km = lon_to_km(latitudes[0], longitudes)

    # Create 2D arrays for the x and y distances
    X_km, Y_km = np.meshgrid(x_km, y_km)

    #Add meter grid to dataset
    dataset['lon_km'] = (('eta', 'xi'), X_km, {'long_name' : 'X-distance from Western boundary', 'units': 'kilometer'})
    dataset['lat_km'] = (('eta', 'xi'), Y_km, {'long_name' : 'Y-distance from Southern boundary', 'units': 'kilometer'})
