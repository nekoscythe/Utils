import numpy as np

def scale_depth(z_levs, z_crit, max_multiplier=1, poly_order=1):
    """
    Scales depth levels based on a critical depth.

    This function applies a scaling factor to depth levels ('z_levs') based on a critical depth
    ('z_crit'). Depths shallower than 'z_crit' are scaled by 'max_multiplier', while deeper
    levels are scaled using a polynomial function to smoothly transition the scaling.

    Parameters:
        z_levs (np.ndarray): Array of depth levels (negative values, increasing downwards).
        z_crit (float): The critical depth (negative value) below which scaling starts to decrease.
        max_multiplier (float, optional): The maximum scaling multiplier applied to shallow depths. Defaults to 1.
        poly_order (int, optional): The order of the polynomial used for scaling deeper depths. Defaults to 1.

    Returns:
        np.ndarray: Array of scaling factors for each depth level.
    """
    z_crit = -1 * z_crit
    z_crit = max(z_levs.min(), z_crit)  # Ensure z_crit is within the range of z_levs

    # Compute scaling factor based on a power curve
    above_crit = (z_levs < z_crit) * max_multiplier
    below_crit = (z_levs >= z_crit) * (1 - (z_crit - z_levs) / z_crit)**poly_order * max_multiplier  # Cubic for sharper acceleration

    return below_crit + above_crit


def get_zlevs(dataset):
    """
    Identifies and returns the z-level variable and its values from the dataset.

    This function checks for common z-level variable names ('z_rho', 'z_w', 'z_rho_u', 'z_rho_v')
    in the dataset and returns the first one found, along with its values for the first time step
    and at the corner grid point (0,0). If no z-level variable is found, it checks for s-level
    variables ('s_rho', 's_w', 's_rho_u', 's_rho_v') and returns the first one found.

    Parameters:
        dataset (xarray.Dataset): Input dataset.

    Returns:
        tuple: A tuple containing:
            - z_levels (np.ndarray): The z-level or s-level array.
            - z_var_name (str): The name of the z-level or s-level variable found.
                                 Returns None, None if no z or s level variable is found.

    Raises:
        Warning: If no z-level variable is found, a warning is printed and s-levels are checked.
                 If no s-level variable is also found, another warning is printed and None, None is returned.
    """
    z_vars = ["z_rho", "z_w", "z_rho_u", "z_rho_v"]
    for z_var in z_vars:
        if z_var in dataset:
            return dataset[z_var][0, :, 0, 0].values, z_var

    print("Warning: No z-level variable found in the dataset.")
    s_vars = ["s_rho", "s_w", "s_rho_u", "s_rho_v"]
    for s_var in s_vars:
        if s_var in dataset:
            print("Returning s-levels instead.")
            return dataset[s_var].values, s_var
    print("Warning: No s-level variable found in the dataset.")
    print("Returning None for z-levels.")
    return None, None


def get_depth_index(z_levs, depth):
    """
    Finds the index in the z-levels array that corresponds to a given depth.

    This function searches for the index in the 'z_levs' array that is closest to the specified
    'depth'. It assumes 'z_levs' are negative values representing depth below the surface.

    Parameters:
        z_levs (np.ndarray): Array of depth levels (negative values, increasing downwards).
        depth (float): Target depth in meters (positive value).

    Returns:
        int: The index in 'z_levs' corresponding to the closest depth level. Returns 0 (bottom index)
             if the target depth exceeds the maximum depth in 'z_levs'.

    Raises:
        Warning: If the target depth exceeds the maximum depth, a warning is printed, and the bottom index is returned.
    """
    z = np.abs(z_levs)
    if depth >= z.max():
        print(f"Warning: Depth {depth} exceeds the maximum depth {z.max()}. Using bottom index.")
        return 0 # Bottom index
    return (z <= depth).argmax() - 1
