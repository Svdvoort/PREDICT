import numpy as np
from skimage.feature import local_binary_pattern
import pandas as pd
import scipy.stats


def get_LBP_features(image, mask, radius, N_points, method):

    feature_names = ['LBP_mean', 'LBP_std', 'LBP_median', 'LBP_kurtosis', 'LBP_skew', 'LBP_peak']

    full_features = np.zeros([len(radius) * len(feature_names), 1])
    full_feature_labels = list()

    mask = mask.flatten()

    for i_index, (i_radius, i_N_points) in enumerate(zip(radius, N_points)):
        LBP_image = np.zeros(image.shape)
        for i_slice in range(0, image.shape[2]):
                LBP_image[:, :, i_slice] = local_binary_pattern(image[:, :, i_slice], P=i_N_points, R=i_radius, method=method)
        LBP_image = LBP_image.flatten()
        LBP_tumor = LBP_image[mask]

        mean_val = np.mean(LBP_tumor)
        std_val = np.std(LBP_tumor)
        median_val = np.median(LBP_tumor)
        kurtosis_val = scipy.stats.kurtosis(LBP_tumor)
        skew_val = scipy.stats.skew(LBP_tumor)
        peak_val = np.bincount(LBP_tumor.astype(np.int))[1:-2].argmax()

        features = [mean_val, std_val, median_val, kurtosis_val, skew_val, peak_val]

        full_feature_start = i_index * len(feature_names)
        full_feature_end = (i_index + 1) * len(feature_names)
        full_features[full_feature_start:full_feature_end, 0] = np.asarray(features).ravel()

        cur_feature_names = [feature_name + '_R' + str(i_radius) + '_P' + str(i_N_points) for feature_name in feature_names]

        full_feature_labels.extend(cur_feature_names)

    features = dict(zip(full_feature_labels, full_features))

    return features


def get_texture_features(image, mask, texture_settings):
    LBP_features = get_LBP_features(image, mask, texture_settings['LBP']['radius'],
                                             texture_settings['LBP']['N_points'],
                                             texture_settings['LBP']['method'])


    texture_features = LBP_features
    texture_features = pd.Series(texture_features)

    return texture_features
