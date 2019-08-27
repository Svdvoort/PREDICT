import configparser
import numpy as np
import re
import os


def load_config(config_file_path):
    """
    Load the config ini, parse settings to PREDICT

    Args:
        config_file_path (String): path of the .ini config file

    Returns:
        settings_dict (dict): dict with the loaded settings
    """

    if os.path.exists(config_file_path):
        settings = configparser.ConfigParser()
        settings.read(config_file_path)
    else:
        raise IOError('Settings file not found!')

    settings_dict = {'DataPaths': dict(), 'CrossValidation': dict(),
                     'Genetics': dict(), 'ImageFeatures': dict(),
                     'HyperOptimization': dict(), 'General': dict(),
                     'Classification': dict(), 'FeatureScaling': dict(),
                     'SampleProcessing': dict()}

    settings_dict['General']['cross_validation'] =\
        settings['General'].getboolean('cross_validation')

    settings_dict['General']['construction_type'] =\
        str(settings['General']['construction_type'])
    settings_dict['General']['N_jobs'] = \
        settings['General'].getint('N_jobs')

    settings_dict['Classification']['classifier'] =\
        str(settings['Classification']['classifier'])

    # First load the datapaths
    settings_dict['DataPaths']['svm_file'] =\
        str(settings['DataPaths']['svm_file'])

    settings_dict['DataPaths']['genetic_file'] =\
        str(settings['DataPaths']['genetic_file'])

    settings_dict['DataPaths']['image_feature_file'] =\
        str(settings['DataPaths']['image_feature_file'])

    # Cross validation settings
    settings_dict['CrossValidation']['N_iterations'] =\
        settings['CrossValidation'].getint('N_iterations')

    settings_dict['CrossValidation']['test_size'] =\
        settings['CrossValidation'].getfloat('test_size')
    settings_dict['CrossValidation']['stratify'] =\
        settings['CrossValidation'].getboolean('stratify')

    # Genetic settings
    mutation_setting = str(settings['Genetics']['mutation_type'])

    mutation_types = re.findall("\[(.*?)\]", mutation_setting)

    for i_index, i_mutation in enumerate(mutation_types):
        stripped_mutation_type = [x.strip() for x in i_mutation.split(',')]
        mutation_types[i_index] = stripped_mutation_type

    settings_dict['Genetics']['mutation_type'] =\
        mutation_types

    settings_dict['Genetics']['genetic_file'] =\
        str(settings['DataPaths']['genetic_file'])

    # Settings for image features
    settings_dict['ImageFeatures']['patient_root_folder'] =\
        str(settings['ImageFeatures']['patient_root_folder'])

    settings_dict['ImageFeatures']['image_locations'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['image_folders'].split(',')]

    settings_dict['ImageFeatures']['mask_files'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['contour_files'].split(',')]

    settings_dict['ImageFeatures']['image_type'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['image_type'].split(',')]

    settings_dict['ImageFeatures']['ROI_files'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['ROI_files'].split(',')]


    settings_dict['ImageFeatures']['image_feature_file'] =\
        str(settings['DataPaths']['image_feature_file'])

    # Groups of features
    settings_dict['ImageFeatures']['feature_settings'] = dict()
    settings_dict['ImageFeatures']['feature_settings']['Patient_features'] = \
        settings['ImageFeatures'].getboolean('patient_features')
    settings_dict['ImageFeatures']['feature_settings']['Location_features'] = \
        settings['ImageFeatures'].getboolean('location_features')
    settings_dict['ImageFeatures']['feature_settings']['Shape_features'] = \
        settings['ImageFeatures'].getboolean('shape_features')
    settings_dict['ImageFeatures']['feature_settings']['Histogram_features'] = \
        settings['ImageFeatures'].getboolean('histogram_features')
    settings_dict['ImageFeatures']['feature_settings']['Texture_features'] = \
        settings['ImageFeatures'].getboolean('texture_features')


    settings_dict['ImageFeatures']['feature_settings']['MR'] = dict()
    settings_dict['ImageFeatures']['feature_settings']['MR']['Normalize'] =\
        [str(item).strip() for item in
         settings['ImageFeatures']['normalize'].split(',')]

    # Patient feature settings
    settings_dict['ImageFeatures']['feature_settings']['Patient'] = dict()
    settings_dict['ImageFeatures']['feature_settings']['Patient']['info_location'] =\
        str(settings['PatientFeatures']['info_location'])
    settings_dict['ImageFeatures']['feature_settings']['Patient']['info_file'] =\
        str(settings['PatientFeatures']['info_file'])

    settings_dict['ImageFeatures']['feature_settings']['Shape_features_3D'] =  settings['ImageFeatures'].getboolean('Shape_features_3D')



    # Shape feature settings
    settings_dict['ImageFeatures']['feature_settings']['shape'] = dict()
    settings_dict['ImageFeatures']['feature_settings']['shape']['multi_mask'] =\
        settings['ShapeFeatures'].getboolean('multi_mask')
    settings_dict['ImageFeatures']['feature_settings']['shape']['mask_index'] = \
        settings['ShapeFeatures'].getint('mask_index')
    settings_dict['ImageFeatures']['feature_settings']['shape']['N_min_smooth'] = \
        settings['ShapeFeatures'].getint('N_min_smooth')
    settings_dict['ImageFeatures']['feature_settings']['shape']['N_max_smooth'] = \
        settings['ShapeFeatures'].getint('N_max_smooth')
    settings_dict['ImageFeatures']['feature_settings']['shape']['min_boundary_points'] = \
        settings['ShapeFeatures'].getint('min_boundary_points')
    # Location feature settins
    settings_dict['ImageFeatures']['feature_settings']['location'] = dict()
    settings_dict['ImageFeatures']['feature_settings']['location']['ROI_index'] =\
        settings['LocationFeatures'].getint('ROI_index')
    settings_dict['ImageFeatures']['feature_settings']['location']['mask_index'] =\
        settings['LocationFeatures'].getint('mask_index')

    # Histogram feature settings
    settings_dict['ImageFeatures']['feature_settings']['histogram'] = dict()
    settings_dict['ImageFeatures']['feature_settings']['histogram']['N_bins'] =\
        settings['HistogramFeatures'].getint('N_bins')

    # Texture feature settings
    settings_dict['ImageFeatures']['feature_settings']['texture'] = dict()

    # LBP settings
    settings_dict['ImageFeatures']['feature_settings']['texture']['LBP'] = dict()
    settings_dict['ImageFeatures']['feature_settings']['texture']['LBP']['radius'] =\
        [int(item) for item in
         settings['TextureFeatures']['LBP_radius'].split(',')]
    settings_dict['ImageFeatures']['feature_settings']['texture']['LBP']['N_points'] =\
        [int(item) for item in
         settings['TextureFeatures']['LBP_points'].split(',')]
    settings_dict['ImageFeatures']['feature_settings']['texture']['LBP']['method'] =\
        str(settings['TextureFeatures']['LBP_method'])


    settings_dict['HyperOptimization']['scoring_method'] =\
        str(settings['HyperOptimization']['scoring_method'])
    settings_dict['HyperOptimization']['test_size'] =\
        settings['HyperOptimization'].getfloat('test_size')
    settings_dict['HyperOptimization']['N_folds'] =\
        settings['HyperOptimization'].getint('N_folds')
    settings_dict['HyperOptimization']['N_search_iter'] =\
        settings['HyperOptimization'].getint('N_search_iter')
    settings_dict['HyperOptimization']['C_loc'] =\
        settings['HyperOptimization'].getfloat('C_loc')
    settings_dict['HyperOptimization']['C_scale'] =\
        settings['HyperOptimization'].getfloat('C_scale')
    settings_dict['HyperOptimization']['degree_loc'] =\
        settings['HyperOptimization'].getfloat('degree_loc')
    settings_dict['HyperOptimization']['degree_scale'] =\
        settings['HyperOptimization'].getfloat('degree_scale')
    settings_dict['HyperOptimization']['coef_loc'] =\
        settings['HyperOptimization'].getfloat('coef_loc')
    settings_dict['HyperOptimization']['coef_scale'] =\
        settings['HyperOptimization'].getfloat('coef_scale')
    settings_dict['HyperOptimization']['gamma_loc'] =\
        settings['HyperOptimization'].getfloat('gamma_loc')
    settings_dict['HyperOptimization']['gamma_scale'] =\
        settings['HyperOptimization'].getfloat('gamma_scale')
    settings_dict['HyperOptimization']['class_weight'] =\
        settings['HyperOptimization']['class_weight']


    settings_dict['FeatureScaling']['scale_features'] =\
        settings['FeatureScaling'].getboolean('scale_features')
    settings_dict['FeatureScaling']['scaling_method'] =\
        str(settings['FeatureScaling']['scaling_method'])

    settings_dict['SampleProcessing']['SMOTE'] =\
        settings['SampleProcessing'].getboolean('SMOTE')

    settings_dict['SampleProcessing']['SMOTE_ratio'] =\
        settings['SampleProcessing'].getfloat('SMOTE_ratio')

    return settings_dict
