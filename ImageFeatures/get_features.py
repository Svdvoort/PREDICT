import numpy as np

import Helpers.sitk_helper as sitkh
import ImageFeatures.histogram_features as hf
import ImageFeatures.texture_features as tf
import ImageFeatures.shape_features as sf
import ImageFeatures.patient_features as pf
import ImageFeatures.location_features as lf
import Helpers.image_helper as ih

# CONSTANTS
N_BINS = 50

def get_image_features(patient_ID, image_data, masks, ROIs,
                       feature_settings):

    N_images = len(image_data.images)
    N_masks = len(masks)
    image_features = dict()

    #~~~~ SHAPE FEATURES ~~~~#

    if feature_settings['Shape_features']:
        multi_mask = feature_settings['shape']['multi_mask']
        mask_index = feature_settings['shape']['mask_index']
        if ~multi_mask and mask_index == -1:
            raise ValueError('Multi_mask was set to False, but no mask index was\
                             provided')

        if multi_mask and N_images != N_masks:
            raise ValueError('Multi_contour was set to True, but the number of\
                             contours does not match the number of images')

        if multi_mask:
            pass
        else:
            shape_mask = ih.get_masked_slices_mask(masks[mask_index])
            shape_features = sf.get_shape_features(shape_mask, feature_settings['shape'])
            image_features['Shape_2D'] = shape_features

            if feature_settings['Shape_features_3D']:
                shape_features_3D = sf.get_3D_shape_features(masks[mask_index])
                image_features['Shape_3D'] = shape_features_3D



    #~~~~~ LOCATION FEATURES ~~~~#
    if feature_settings['Location_features']:
        ROI_index = feature_settings['location']['ROI_index']
        mask_index = feature_settings['location']['mask_index']
        location_features = lf.get_location_features(masks[mask_index],
                                                     ROIs[ROI_index])
        image_features['Location'] = location_features


    images = image_data['images']
    image_types = image_data['images'].keys()
    meta_data = image_data['metadata']


    if feature_settings['Patient_features']:
        if feature_settings['Patient']['info_location'] == 'dicom':
            patient_features = pf.get_patient_features(meta_data[0])

        elif feature_settings['Patient']['info_location'] == 'file':
            patient_file = feature_settings['Patient']['info_file']
            patient_features = pf.get_patient_features_file(patient_ID,
                                                            patient_file)

        image_features['patient_features'] = patient_features

    for i_image, i_mask, i_image_type, i_meta_data in zip(images, masks,
                                                          image_types,
                                                          meta_data):

        if 'MR' in i_image_type:
            i_image_array = sitkh.GetArrayFromImage(i_image)
            i_mask_array = sitkh.GetArrayFromImage(i_mask)

            if i_image_array.shape != i_mask_array.shape:
                raise ValueError('Mask and image do not match!')

            if feature_settings['Histogram_features']:
                N_bins = feature_settings['histogram']['N_bins']
                masked_voxels = ih.get_masked_voxels(i_image_array, i_mask_array)

                histogram_features = hf.get_histogram_features(masked_voxels,
                                                               N_bins)

                image_features[i_image_type + '_histogram_features'] = \
                    histogram_features

            if feature_settings['Texture_features']:
                i_image_array, i_mask_array = ih.get_masked_slices_image(
                    i_image_array, i_mask_array)

                texture_features = tf.get_texture_features(i_image_array,
                                                           i_mask_array,
                                                           feature_settings['texture'])
                image_features[i_image_type + '_texture_features'] =\
                    texture_features

    # We also return just the arrray
    image_feature_array = list()

    for _, feature in image_features.iteritems():
        image_feature_array.extend(feature.values)

    image_feature_array = np.asarray(image_feature_array)
    image_feature_array = image_feature_array.ravel()

    return image_features, image_feature_array
