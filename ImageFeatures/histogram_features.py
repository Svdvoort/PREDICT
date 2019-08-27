import numpy as np
import scipy.stats
import pandas as pd

import Helpers.contour_functions as cf


def create_histogram(data, bins):
    histogram, bins = np.histogram(data, bins)
    return histogram, bins


def get_min(data):
    # return np.amin(data)
    return np.percentile(data, 2)


def get_max(data):
    # return np.amax(data)
    return np.percentile(data, 98)


def get_median(data):
    return np.median(data)


def get_mean(data):
    return np.mean(data)


def get_std(data):
    return np.std(data)


def get_skewness(data):
    return scipy.stats.skew(data)


def get_kurtosis(data):
    return scipy.stats.kurtosis(data)


def get_peak_position(histogram, bins):
    return np.amax(histogram)


def get_diff_in_out(image, contour):
    _, voi_voxels = cf.get_voi_voxels(contour, image)
    _, not_voi_voxels = cf.get_not_voi_voxels(contour, image)
    return np.mean(voi_voxels) - np.mean(not_voi_voxels)


def get_range(data):
    return np.percentile(data, 98) - np.percentile(data, 2)


def get_energy(data):
    energy = np.sum(np.square(data + np.min(data)))
    return energy


def get_quartile_range(data):
    return np.percentile(data, 75) - np.percentile(data, 25)


def get_histogram_features(data, N_bins):
    hist_min = get_min(data)
    hist_max = get_max(data)
    hist_mean = get_mean(data)
    hist_median = get_median(data)
    hist_std = get_std(data)
    hist_skewness = get_skewness(data)
    hist_kurtosis = get_kurtosis(data)
    hist_range = get_range(data)
    temp_histogram, temp_bins = create_histogram(data, N_bins)
    hist_peak = get_peak_position(temp_histogram, temp_bins)
    energy = get_energy(data)
    quartile_range = get_quartile_range(data)

    panda_labels = ['hist_skewness', 'hist_kurtosis', 'hist_mean', 'hist_median', 'hist_max',
                    'hist_std', 'hist_min', 'hist_range', 'hist_peak', 'energy',
                    'quartile_range']

    histogram_features = [hist_skewness, hist_kurtosis, hist_mean, hist_median, hist_max,
                          hist_std, hist_min, hist_range, hist_peak, energy,
                          quartile_range]




    panda_dict = dict(zip(panda_labels, histogram_features))

    histogram_features = pd.Series(panda_dict)

    return histogram_features
