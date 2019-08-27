import numpy as np
import pandas as pd

import ImageFeatures.get_features as gf
import SimpleITK as sitk
import os

# There is a small difference between the mask and image origin and spacing
# Fix this by setting a slightly larger, but still reasonable tolerance
# (Defaults to around 8e-7, which seems very small)
sitk.ProcessObject.SetGlobalDefaultCoordinateTolerance(1e-1)
sitk.ProcessObject.SetGlobalDefaultDirectionTolerance(1e-1)

def get_image_features(patient_root_folder, image_feature_file,
                       image_locations, image_type, mask_files,
                       ROI_files, feature_settings, patient_IDs):

    """Computes (or if already available loads) the image features from the
     given images and settings

    Args:
        patient_root_folder (string): Path containing the individual patient
        folders
        image_feature_file (string): Basename for the feature file
        image_locations (list): List of folders and filenames of images to
        process
        image_type (list): List of strings with the type of image
        mask_files (list): Location of masks for feature extraction
        ROI_files (list): Location of general ROIs (for e.g. relative location)
        feature_settings (dict): Settings that will be used for feature
        calculations
        patient_IDs(list): patients that will be processed.
    Returns:
        dict: image_features (array): Contains the values of the image
        features for all patients
    """

    panda_labels = ['patient_root_folder', 'genetic_file',
                    'image_feature_file', 'image_folders',
                    'image_type', 'mask_files', 'feature_settings',
                    'patient_ID', 'image_features',
                    'image_features_array']

    image_features = list()
    for i_patient in patient_IDs:
        i_patient = str(i_patient)
        print('Now processing: ' + str(i_patient))

        pandas_file_name = image_feature_file + '_' +\
            i_patient + '.hdf5'

        if os.path.isfile(pandas_file_name):
            print('Found image features!')
            panda_data_loaded = pd.read_hdf(pandas_file_name)
            image_features_array = panda_data_loaded.image_features_array
        else:
            print('Calculating image features!')

            patient_folder = os.path.join(patient_root_folder, i_patient)

            ROIs = load_ROIs(patient_folder, ROI_files)


            image_data = load_images(i_patient, patient_folder,
                                     image_locations, image_type,
                                     feature_settings, ROIs)

            masks = load_masks(patient_folder, mask_files)


            image_features_temp, image_features_array =\
                gf.get_image_features(i_patient, image_data, masks, ROIs,
                                      feature_settings)

            panda_data = pd.Series([patient_root_folder, image_feature_file,
                                    image_locations, image_type, mask_files,
                                    ROI_files, feature_settings,
                                    i_patient, image_features_temp,
                                    image_features_array],
                                   index=panda_labels,
                                   name='Image features'
                                   )

            print('Saving image features')
            panda_data.to_hdf(pandas_file_name, 'image_features')

        if any(np.isnan(image_features_array)) or\
           any(np.isinf(image_features_array)):
            err_text = 'Patient ' + i_patient + ' has invalid feature values!'
            print(image_features_temp)
            print(image_features_array)
            raise ValueError(err_text)
        else:
            image_features.append(image_features_array)

    image_features = np.asarray(image_features)

    return image_features


def load_images(patient_ID, patient_folder, image_folders, image_type,
                feature_settings, ROIs):
    images = list()
    metadata = list()

    for i_image_folder, i_image_type, i_normalization, i_ROI in zip(image_folders, image_type,  feature_settings['MR']['Normalize'], ROIs):
        image_folder = os.path.join(patient_folder, i_image_folder)
        if 'MR' in i_image_type:
            image_temp = sitk.ReadImage(image_folder)
            metadata_temp = None
        if i_normalization == 'z_score':
            image_temp = sitk.Normalize(image_temp)
        elif i_normalization == 'ROI':
            # Cast to float to allow proper processing
            image_temp = sitk.Cast(image_temp, 9)

            LabelFilter = sitk.LabelStatisticsImageFilter()
            LabelFilter.Execute(image_temp, i_ROI)
            ROI_mean = LabelFilter.GetMean(1)
            ROI_std = LabelFilter.GetSigma(1)

            image_temp = sitk.ShiftScale(image_temp,
                                         shift=-ROI_mean,
                                         scale=1.0/ROI_std)
        else:
            raise ValueError('Unknown normalization option!')

        images.append(image_temp)
        metadata.append(metadata_temp)

    image_data = pd.DataFrame({'images': images,
                               'metadata': metadata},
                              index=image_type)

    return image_data


def load_masks(patient_folder, mask_files):
    """
    Loads masks

    Args:
        patient_folder (string): the base patient folder
        mask_files (list): List of strings of names of masks
    Returns:
        masks (list): The loaded masks
    """

    masks = list()
    for i_mask_file in mask_files:
        mask_file = os.path.join(patient_folder, i_mask_file)
        mask_image = sitk.ReadImage(mask_file)
        # Need binary mask
        mask_image = sitk.Cast(mask_image, 0)
        masks.append(mask_image)
    return masks


def load_ROIs(patient_folder, ROI_files):
    ROIs = list()
    for i_ROI_file in ROI_files:
        ROI_file = os.path.join(patient_folder, i_ROI_file)
        if os.path.exists(ROI_file):
            ROI_image = sitk.ReadImage(ROI_file)
            ROI_image = sitk.Cast(ROI_image, 0)
            ROIs.append(ROI_image)
        else:
            ROIs.append(None)
    return ROIs
