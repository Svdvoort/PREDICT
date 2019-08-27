import SVM.crossvalidate_SVM as crossval_SVM


def construct_classifier(config, mutation_data, image_features):
    """Interface to create classification

    Different classifications can be created using this common interface

    Args:
        config (dict): Dictionary of the required config settings
        mutation_data (dict): Mutation data that should be classified
        features (pandas dataframe): A pandas dataframe containing the features
         to be used for classification

    Returns:
        Constructed classifier
    """

    if config['Classification']['classifier'] == 'SVM':
        classifier = construct_SVM(config, mutation_data, image_features)

    return classifier


def construct_SVM(config, mutation_data, image_features):
    """
    Constructs a SVM classifier

    Args:
        config (dict): Dictionary of the required config settings
        mutation_data (dict): Mutation data that should be classified
        features (pandas dataframe): A pandas dataframe containing the features
         to be used for classification

    Returns:
        SVM classifier
    """

    patient_IDs = mutation_data['patient_IDs']
    mutation_label = mutation_data['mutation_label']
    mutation_name = mutation_data['mutation_name']

    if config['General']['cross_validation']:
        if config['General']['construction_type'] == 'BinaryClass':
            # Construct just a normal SVM
            SVM_data =\
              crossval_SVM.construct_binary_svm(config, patient_IDs,
                                                  image_features,
                                                  mutation_label,
                                                  mutation_name)

    elif config['General']['construction_type'] == 'All':
        # Construct SVM without cross-validation
        SVM_data =\
            crossval_SVM.construct_all_svm(config, patient_IDs,
                                           image_features,
                                           mutation_label, mutation_name)

    return SVM_data
