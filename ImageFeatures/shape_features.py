import numpy as np
import pandas as pd
import Helpers.contour_functions as cf
import SimpleITK as sitk


def get_shape_features(mask, shape_settings):
    # CONSTANTS
    N_min_smooth = shape_settings['N_min_smooth']
    N_max_smooth = shape_settings['N_max_smooth']
    min_boundary_points = shape_settings['min_boundary_points']

    N_mask_slices = mask.GetSize()[2]

    # Pre-allocation
    perimeter = np.zeros([N_mask_slices, 1])
    convexity = np.zeros([N_mask_slices, 1])
    area = np.zeros([N_mask_slices, 1])
    rad_dist_avg = np.zeros([N_mask_slices, 1])
    rad_dist_std = np.zeros([N_mask_slices, 1])
    roughness_avg = np.zeros([N_mask_slices, 1])
    roughness_std = np.zeros([N_mask_slices, 1])
    cvar = np.zeros([N_mask_slices, 1])
    prax = np.zeros([N_mask_slices, 1])
    evar = np.zeros([N_mask_slices, 1])
    solidity = np.zeros([N_mask_slices, 1])
    compactness = np.zeros([N_mask_slices, 1])

    rad_dist = list()
    rad_dist_norm = list()
    roughness = list()

    # Now calculate some of the edge shape features
    # TODO: Adapt to allow for multiple slices at once
    for i_slice in range(0, N_mask_slices):
        boundary_points = cf.get_smooth_contour(mask[:, :, i_slice],
                                                N_min_smooth,
                                                N_max_smooth)

        if boundary_points is None or boundary_points.shape[0] <= min_boundary_points:
            # Only 1 or 2 points in volume, which means it's not really a
            # volume, therefore we ignore it.
            continue

        temp_area = compute_area(boundary_points)

        if temp_area == 0:
            continue

        rad_dist_i, rad_dist_norm_i = compute_radial_distance(
            boundary_points)
        rad_dist.append(rad_dist_i)
        rad_dist_norm.append(rad_dist_norm_i)
        perimeter[i_slice] = compute_perimeter(boundary_points)

        area[i_slice] = temp_area
        compactness[i_slice] = compute_compactness(boundary_points)
        roughness_i, roughness_avg[i_slice] = compute_roughness(
            boundary_points, rad_dist_i)
        roughness.append(roughness_i)

        cvar[i_slice] = compute_cvar(boundary_points)
        prax[i_slice] = compute_prax(boundary_points)
        evar[i_slice] = compute_evar(boundary_points)

        # TODO: Move computing convexity into esf
        convex_hull = cf.convex_hull(mask[:, :, i_slice])
        convexity[i_slice] = compute_perimeter(convex_hull)\
            / perimeter[i_slice]

        solidity[i_slice] = compute_area(convex_hull)/area[i_slice]
        rad_dist_avg[i_slice] = np.mean(np.asarray(rad_dist_i))
        rad_dist_std[i_slice] = np.std(np.asarray(rad_dist_i))
        roughness_std[i_slice] = np.std(np.asarray(roughness_i))

    compactness_avg = np.mean(compactness)
    compactness_std = np.std(compactness)
    convexity_avg = np.mean(convexity)
    convexity_std = np.std(convexity)
    rad_dist_avg = np.mean(rad_dist_avg)
    rad_dist_std = np.mean(rad_dist_std)
    roughness_avg = np.mean(roughness_avg)
    roughness_std = np.mean(roughness_std)
    cvar_avg = np.mean(cvar)
    cvar_std = np.std(cvar)
    prax_avg = np.mean(prax)
    prax_std = np.std(prax)
    evar_avg = np.mean(evar)
    evar_std = np.std(evar)
    solidity_avg = np.mean(solidity)
    solidity_std = np.std(solidity)

    panda_labels = ['compactness_avg', 'compactness_std', 'rad_dist_avg', 'rad_dist_std',
                    'roughness_avg', 'roughness_std',
                    'convexity_avg', 'convexity_std', 'cvar_avg', 'cvar_std',
                    'prax_avg', 'prax_std', 'evar_avg', 'evar_std', 'solidity_avg', 'solidity_std']

    shape_features = [compactness_avg, compactness_std, rad_dist_avg, rad_dist_std,
                      roughness_avg, roughness_std,
                      convexity_avg, convexity_std, cvar_avg, cvar_std,
                      prax_avg, prax_std, evar_avg, evar_std, solidity_avg, solidity_std]

    pandas_dict = dict(zip(panda_labels, shape_features))
    shape_features = pd.Series(pandas_dict)

    return shape_features


def get_3D_shape_features(mask):
    ShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
    ShapeStatistics.Execute(mask)

    mask_volume = ShapeStatistics.GetPhysicalSize(1)


    # Selected features
    panda_labels = ['Volume']
    shape_features_3D = [mask_volume]


    pandas_dict = dict(zip(panda_labels, shape_features_3D))
    shape_features_3D = pd.Series(pandas_dict)

    return shape_features_3D


def get_center(points):
    """Computes the center of the given boundary points"""
    x_center = np.mean(points[:, 0])
    y_center = np.mean(points[:, 1])
    return x_center, y_center


def compute_dist_to_center(points):
    """Computes the distance to the center for the given boundary points"""
    center = get_center(points)

    dist_to_center = points - center
    return dist_to_center


def compute_abs_dist_to_center(points):
    """Computes the absolute distance to center for given boundary points"""

    dist_center = compute_dist_to_center(points)
    abs_dist_to_center = np.sqrt(dist_center[:, 0]**2 + dist_center[:, 1]**2)

    return abs_dist_to_center


def compute_perimeter(points):
    """Computes the perimeter of the given boundary points"""
    xdiff = np.diff(points[:, 0])
    ydiff = np.diff(points[:, 1])

    perimeter = np.sum(np.sqrt(xdiff**2 + ydiff**2))

    return perimeter


def compute_area(points):
    """Computes the area of the given boundary points using shoelace formula"""
    x = points[:, 0]
    y = points[:, 1]

    area = 0.5*np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    return area


def compute_compactness(points):
    """Computes compactness of the given boundary points, 1 for circle"""

    perimeter = compute_perimeter(points)
    area = compute_area(points)

    compactness = 4*np.pi*(area/perimeter**2)

    return compactness


def compute_radial_distance(points):
    """
    Computes the radial distance for the given boundary points, according to
    Xu et al 2012, "A comprehensive descriptor of shape"

    """
    dist_center = compute_dist_to_center(points)
    rad_dist = np.sqrt(dist_center[:, 0]**2 + dist_center[:, 1]**2)
    rad_dist_norm = rad_dist/np.amax(rad_dist)

    return rad_dist, rad_dist_norm


def compute_roughness(points, rad_distance=None, min_points=3, max_points=15):
    """
    compute_roughness computes the roughness according to "Xu et al. 2012,
    A comprehensive descriptor of shape"

    Args:
        points ([Nx2] numpy array): array of boundary points

    Kwargs:
        rad_distance (numpy array): Radial distance if already computed
                                    [default: None]
        min_points (int): Minimum number of points in a segment
                          [default: 3]
        max_points (int): Maximum number of points in a segment
                          [default: 15]

    Returns:
        roughness (numpy array): The roughness in the different segments
        roughness_avg (float): The average roughness

    """
    if rad_distance is None:
        rad_distance = compute_radial_distance(points)

    N_points = points.shape[0]

    # Find the number of points in a segment, by looking for number that will
    # perfectly divide the boundary into equal segements
    N_points_segment = min_points
    while N_points % N_points_segment != 0 and N_points_segment < max_points:
        N_points_segment += 1

    if N_points_segment == max_points:
        # Not perfectly divisble, so round down
        N_segments = np.floor(N_points/N_points_segment)
    else:
        N_segments = N_points/N_points_segment

    N_segments = int(N_segments)

    roughness = np.zeros([N_segments, 1])

    for i_segment in range(0, N_segments):
        if i_segment == N_segments - 1:
            # If the number of segments is not a perfect fit, the last segment
            # gets all the leftover points
            cur_segment = range(i_segment*N_points_segment, N_points)
        else:
            cur_segment = range(i_segment*N_points_segment,
                                (i_segment+1)*N_points_segment)

        roughness[i_segment] = np.sum(np.abs(np.diff(
                                                rad_distance[cur_segment])))

    roughness_avg = 1.0*N_points_segment/N_points*np.sum(roughness)

    return roughness, roughness_avg


def compute_mean_radius(points):
    """
    Computes mean radius for giving boundary points, according to Peura et al.
    1997, "Efficiency of Simple Shape Descriptors"

    """
    abs_dist_center = compute_abs_dist_to_center(points)
    mean_radius = 1.0*np.sum(abs_dist_center)/points.shape[0]

    return mean_radius


def compute_cvar(points):
    """
    Computes circular variance for giving boundary points, according to
    Peura et al. 1997, "Efficiency of Simple Shape Descriptors"

    """
    abs_dist_center = compute_abs_dist_to_center(points)

    mean_radius = compute_mean_radius(points)

    cvar = 1.0/(points.shape[0]*mean_radius**2)*np.sum((abs_dist_center -
                                                        mean_radius)**2)

    return cvar


def compute_covariance_matrix(points):
    """
    Computes covariance matrix for giving boundary points, according to
    Peura et al. 1997, "Efficiency of Simple Shape Descriptors"

    """
    dist_to_center = compute_dist_to_center(points)

    covariance_matrix = list()
    for i_point in dist_to_center:
        covariance_matrix.append(np.outer(i_point, i_point))

    covariance_matrix = np.asarray(covariance_matrix)
    covariance_matrix = 1.0*np.sum(covariance_matrix, 0)/points.shape[0]

    return covariance_matrix


def compute_prax(points):
    """
    Computes ratio of principal axes for giving boundary points, according to
    Peura et al. 1997, "Efficiency of Simple Shape Descriptors"

    """
    covariance_matrix = compute_covariance_matrix(points)

    c_xx = covariance_matrix[0][0]
    c_yy = covariance_matrix[1][1]
    c_xy = covariance_matrix[0][1]

    first_term = c_xx + c_yy
    second_term = np.sqrt((c_xx + c_yy)**2 - 4*(c_xx*c_yy - c_xy**2))

    prax = 1.0*(first_term - second_term)/(first_term + second_term)

    return prax


def compute_evar(points):
    """
    Computes eliptic variance for giving boundary points, according to
    Peura et al. 1997, "Efficiency of Simple Shape Descriptors"

    """
    covariance_matrix = compute_covariance_matrix(points)
    dist_to_center = compute_dist_to_center(points)

    if covariance_matrix[0][0] == 0 or covariance_matrix[1][1] == 0:
        # It isn't a well-defined contour
        return 0

    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    root_term = list()

    for i_point in dist_to_center:
        first_product = np.dot(i_point, inv_covariance_matrix)
        second_product = np.dot(first_product, i_point)
        root_term.append(np.sqrt(second_product))

    root_term = np.asarray(root_term)

    mu_rc = 1.0/points.shape[0]*np.sum(root_term)

    evar = 1.0/(points.shape[0]*mu_rc)*np.sum((root_term - mu_rc)**2)

    return evar
