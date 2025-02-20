import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def convolve2d(data, kernel, stride=(1, 1)):
    # Ensure kernel and stride are valid
    if len(kernel.shape) != 2:
        raise ValueError("Kernel must be a 2D array.")
    if len(stride) != 2:
        raise ValueError("Stride must be a tuple of two integers.")

    kernel_height, kernel_width = kernel.shape
    stride_y, stride_x = stride

    # Get dimensions of the input data
    *batch_dims, data_height, data_width = data.shape

    # Calculate output dimensions
    output_height = (data_height - kernel_height) // stride_y + 1
    output_width = (data_width - kernel_width) // stride_x + 1

    # Initialize output array
    output_shape = (*batch_dims, output_height, output_width)
    output = np.zeros(output_shape)

    # Perform convolution
    for y in range(0, output_height):
        for x in range(0, output_width):
            y_start = y * stride_y
            x_start = x * stride_x
            
            region = data[..., y_start:y_start + kernel_height, x_start:x_start + kernel_width]
            output[..., y, x] = np.sum(region * kernel, axis=(-2, -1))

    return output


def coarsen_rho_grid(var_data, n=4):
    #remove the ghost points
    var_data = var_data[..., 1:-1, 1:-1]
    #use a 2d nxn convolution to average the data
    filter = np.ones((n,n))/(n**2) 
    coarsened_data = convolve2d(var_data, filter, stride=(n, n))
    
    #add back the ghost points
    #periodic boundary conditions along x
    #copy 1st column to the end and last column to the beginning
    coarsened_data = np.concatenate([coarsened_data[..., -1:], coarsened_data, coarsened_data[..., :1]], axis=-1)
    
    #no-flux boundary conditions along y
    #copy the 1st row to the beginning and last row to the end
    coarsened_data = np.concatenate([coarsened_data[..., :1,:], coarsened_data, coarsened_data[..., -1:,:]], axis=-2)
    
    return coarsened_data


def coarsen_u_grid(var_data, n=4):
    # Remove ghost points and repeated column for periodicity
    var_data = var_data[..., 1:-1, 1:]
    # Copy the index 1 and index 2 columns to the end
    var_data = np.concatenate([var_data[..., 1:], var_data[..., :2]], axis=-1)
    
    # Vertical edge filter
    filter = np.ones((1,n+1))
    filter[0, 0] = 0.5
    filter[0, -1] = 0.5
    # Repeat the filter to make it 2D
    filter = np.repeat(filter, n, axis=0)
    # Normalize the filter
    filter = filter / np.sum(filter)
    # Convolve the data with the filter, with a stride of 4
    coarsened_data = convolve2d(var_data, filter, stride=(4, 4))
    
    # Add back the ghost points
    #periodic boundary conditions along x
    #copy last column to the beginning
    coarsened_data = np.concatenate([coarsened_data[..., -1:], coarsened_data], axis=-1)
    
    #no-flux boundary conditions along y
    #copy the 1st row to the beginning and last row to the end
    coarsened_data = np.concatenate([coarsened_data[..., :1,:], coarsened_data, coarsened_data[..., -1:,:]], axis=-2)
    return coarsened_data


def coarsen_v_grid(var_data, n=4):
    # Remove ghost points and repeated row for periodicity
    var_data = var_data[..., 1:, 1:-1]
    # Copy the index 1 and index 2 rows to the end
    var_data = np.concatenate([var_data[..., 1:, :], var_data[..., :2, :]], axis=-2)
    
    # Horizontal edge filter
    filter = np.ones((n+1,1))
    filter[0] = 0.5
    filter[-1] = 0.5
    # Repeat the filter to make it 2D
    filter = np.repeat(filter, n, axis=1)
    # Normalize the filter
    filter = filter / np.sum(filter)
    # Convolve the data with the filter, with a stride of 4
    coarsened_data = convolve2d(var_data, filter, stride=(4, 4))
    
    
    # Add back the ghost points
    
    #periodic along x, so copy last column to the beginning and 1st column to the end
    coarsened_data = np.concatenate([coarsened_data[..., -1:], coarsened_data, coarsened_data[..., :1]], axis=-1)
    
    # At N-S boundaries, copy 1st row to the beginning
    coarsened_data = np.concatenate([coarsened_data[..., :1,:], coarsened_data], axis=-2)
    
    return coarsened_data


def rho2uvp(grid):
    """Convert rho points to u, v, and psi points."""
    u = 0.5 * (grid[:, 1:] + grid[:, :-1])  # Average in the xi-direction (columns)
    v = 0.5 * (grid[1:, :] + grid[:-1, :])  # Average in the eta-direction (rows)
    p = 0.5 * (u[1:, :] + u[:-1, :])  # Average of u in the eta-direction (rows)
    return u, v, p


def get_next_lat(lat, dl):
    return lat + dl * np.cos(np.radians(lat))

def get_prev_lat(lat, dl, tolerance=1e-6, max_iterations=100):
    # Initial guess for prev_lat (start with the current latitude)
    prev_lat = lat

    for _ in range(max_iterations):
        # Compute f(prev_lat)
        f = prev_lat + dl * np.cos(np.radians(prev_lat)) - lat

        # Compute f'(prev_lat)
        f_prime = 1 - dl * np.sin(np.radians(prev_lat)) * (np.pi / 180)

        # Update prev_lat using Newton-Raphson
        prev_lat_new = prev_lat - f / f_prime

        # Check for convergence
        if abs(prev_lat_new - prev_lat) < tolerance:
            return prev_lat_new

        prev_lat = prev_lat_new

    # If max_iterations is reached, return the last computed value
    return prev_lat

def latlon_ghost_points(latr, lonr):
    #takes already coarsened latr and lonr arrays and adds the ghost points properly
    #remove the ghost points and convert back to 1d arrays
    latr = latr[1:-1,0]
    lonr = lonr[0, 1:-1]
    
    lonr_diff = np.diff(lonr)
    dl = np.average(lonr_diff)
    lonr = np.concatenate([[lonr[0]-dl], lonr, [lonr[-1]+dl]])
    
    latr = np.concatenate([[get_prev_lat(latr[0], dl)], latr, [get_next_lat(latr[-1], dl)]])
    
    lonr, latr = np.meshgrid(lonr, latr)
    return latr, lonr


def coarsen_dataset(ds, n=4):
     #get all the different grid variables
        xi_dims = [d for d in ds.dims if 'xi_' in d]
        eta_dims = [d for d in ds.dims if 'eta_' in d]
        
        #get all combinations of the grid variables
        grid_vars = []
        for lat in eta_dims:
            for lon in xi_dims:
                grid_vars.append((lat, lon))
                
        #set all unused coords to data_vars
        ds = ds.reset_coords(drop=False)
        
        #create an empty dataset to store the coarsened data
        new_ds = xr.Dataset()
        # iterate over all the grid variables
        for i, (eta, xi) in enumerate(grid_vars):
            #get all the variables that have the grid variables
            var_names = [v for v in ds.data_vars if eta in ds[v].dims and xi in ds[v].dims]
            if len(var_names) == 0:
                continue
            coords = [v for v in ds.coords if (eta in ds[v].dims and xi in ds[v].dims)]
            subset = ds[var_names]
            #iterate over all the variables
            for var in var_names:
                #get the coarsening function based on the grid variable
                coarsen_func = coarsen_v_grid if eta == 'eta_v' else coarsen_u_grid if xi == 'xi_u' else coarsen_rho_grid if eta == 'eta_rho' and xi == 'xi_rho' else None
                if coarsen_func is None:
                    continue
                #coarsen the data
                coarsened_data = coarsen_func(subset[var].values, n)
                #add the coarsened data to the new dataset
                new_ds[var] = (subset[var].dims, coarsened_data)
                # copy the attributes
                new_ds[var].attrs = subset[var].attrs
            
            new_ds[eta] = new_ds[eta].astype(np.float64)
            new_ds[xi] = new_ds[xi].astype(np.float64)
            #copy the attributes
            new_ds[eta].attrs = ds[eta].attrs
            new_ds[xi].attrs = ds[xi].attrs
    
    
        #get all the remaining non-grid variables
        non_grid_vars = [v for v in ds.data_vars if v not in new_ds.data_vars]
        for var in non_grid_vars:
            new_ds[var] = ds[var]
            
        #add all the attributes
        new_ds.attrs = ds.attrs
        
        #fix xi_u and eta_v
        xi_u,_,_ = rho2uvp(np.reshape(new_ds['xi_rho'].values, (1, -1)))
        _,eta_v,_ = rho2uvp(np.reshape(new_ds['eta_rho'].values, (-1, 1)))
        new_ds = new_ds.assign_coords({'xi_u': xi_u[0], 'eta_v': eta_v[:,0]})
        new_ds['xi_u'].attrs = ds['xi_u'].attrs
        new_ds['eta_v'].attrs = ds['eta_v'].attrs
        
        #fix the latitudes and longitudes
        latr, lonr = latlon_ghost_points(new_ds['lat_rho'].values, new_ds['lon_rho'].values)
        latu,latv, _ = rho2uvp(latr)
        lonu,lonv, _ = rho2uvp(lonr)
        
        latlons = {'lat_rho': latr, 'lon_rho': lonr, 'lat_u': latu, 'lon_u': lonu, 'lat_v': latv, 'lon_v': lonv}
        
        for latlon in latlons:
            new_ds[latlon].values = latlons[latlon]
            
        new_ds = new_ds.set_coords(list(latlons.keys()))
            
        
        
        return new_ds
    
    
    
def plot_grid(coarse_grid, m=4, n=4, alpha=1, color='r', linewidth=1.2, s=64, no_ghost_points=False):
    """
    Plot fine and coarse grids with their respective points using line segments for efficiency.

    Parameters:
    - hires_grid: The high-resolution grid data.
    - coarse_grid: The coarsened grid data.
    - m: Number of rows in the subgrid.
    - n: Number of columns in the subgrid.
    - alpha: Transparency level for the fine grid.
    """
    def rho2uvp(grid):
        """Convert rho points to u, v, and psi points."""
        u = 0.5 * (grid[:, 1:] + grid[:, :-1])  # Average in the xi-direction (columns)
        v = 0.5 * (grid[1:, :] + grid[:-1, :])  # Average in the eta-direction (rows)
        p = 0.5 * (u[1:, :] + u[:-1, :])  # Average of u in the eta-direction (rows)
        return u, v, p
    # Extract coarse grid coordinates
    lonr, latr = coarse_grid['lon_rho'].values, coarse_grid['lat_rho'].values
    # Lonu, Lonv, _ = (coarse_grid['lon_u'], coarse_grid['lon_v'], None)
    # Latu, Latv, _ = rho2uvp(latr) if 'lat_u' not in coarse_grid else (coarse_grid['lat_u'], coarse_grid['lat_v'], None) 
    Lonu, Lonv, _ = (coarse_grid['lon_u'], coarse_grid['lon_v'], None)
    Latu, Latv, _ = (coarse_grid['lat_u'], coarse_grid['lat_v'], None) 

    # Extract subgrid range
    k = int(no_ghost_points)  # Exclude ghost points if needed
    Lonr, Latr = lonr[:m+1-k, :n+1-k], latr[:m+1-k, :n+1-k]
    Lonu, Latu = Lonu[:m+1, :n+1], Latu[:m+1, :n+1]  # Align dimensions
    Lonv, Latv = Lonv[:m+1, :n+1], Latv[:m+1, :n+1]  # Align dimensions


    # Function to plot gridlines
    def plot_grid_lines(lon, lat, color, alpha=1.0, linewidth=1.0, zorder=1):
        # Horizontal lines
        for i in range(lat.shape[0]):
            plt.plot(lon[i, :], lat[i, :], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
        # Vertical lines
        for j in range(lon.shape[1]):
            plt.plot(lon[:, j], lat[:, j], color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)


    # Plot coarse grid lines
    plot_grid_lines(Lonu, Latv, color=color, alpha=alpha, linewidth=linewidth, zorder=1)
    
    # Scatter plots for points (set higher zorder to appear on top of lines)
    plt.scatter(Lonr, Latr, marker="x", color=color, s=s, zorder=3)
    if k==1:
        Lonu, Latu = Lonu[:-1], Latu[:-1]
        Lonv, Latv = Lonv[:,:-1], Latv[:,:-1]
    plt.scatter(Lonu, Latu, marker=r"$\rightarrow$", s=s, color='k', zorder=3)
    plt.scatter(Lonv, Latv, marker=r"$\uparrow$", s=s, color='k', zorder=3)
    
