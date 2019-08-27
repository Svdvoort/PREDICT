import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import parameter_optimization as po

from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

from sklearn.utils import check_random_state


def construct_binary_svm(config, patient_IDs, image_features,
                       mutation_label, mutation_name):
    """
    Constructs multiple individual SVMs based on the mutation settings

    Arguments:
        config (Dict): Dictionary with config settings
        patient_IDs (list): IDs of the patients, used to keep track of test and
                     training sets, and genetic data
        image_features (numpy array): The values for the different features
        mutation_label (list): List of lists, where each list contains the
                                mutations status for that patient for each
                                mutations
        mutation_name (list): Contains the different mutations that are stored
                              in the mutation_label


    Returns:
        SVM_data (pandas dataframe)
    """

    N_iterations = config['CrossValidation']['N_iterations']
    test_size = config['CrossValidation']['test_size']
    N_jobs = config['General']['N_jobs']

    svm_mutations = dict()

    print('features')
    print(image_features.shape)
    for i_mutation, i_name in zip(mutation_label, mutation_name):
        i_mutation = i_mutation.ravel()
        i_mutation = i_mutation.astype(np.int)

        save_data = list()

        for i in range(0, N_iterations):
            random_seed = np.random.randint(1, 5000)
            random_state = check_random_state(random_seed)


            print('Iteration:    ' + str(i))

            # Split into test and training set, where the percentage of each
            # label is maintained
            if config['CrossValidation']['stratify']:
                X_train, X_test, Y_train, Y_test,\
                    patient_ID_train, patient_ID_test\
                    = train_test_split(image_features, i_mutation, patient_IDs,
                                       test_size=test_size, random_state=random_state,
                                       stratify=i_mutation)
            else:
                X_train, X_test, Y_train, Y_test,\
                    patient_ID_train, patient_ID_test\
                    = train_test_split(image_features, i_mutation, patient_IDs,
                                       test_size=test_size, random_state=random_state)

            # Scale the features
            if config['FeatureScaling']['scale_features']:
                if config['FeatureScaling']['scaling_method'] == 'z_score':
                    scaler = StandardScaler().fit(X_train)
                elif config['FeatureScaling']['scaling_method'] == 'minmax':
                    scaler = MinMaxScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)


            if config['SampleProcessing']['SMOTE']:
                sm = SMOTE(random_state=random_state,
                           ratio=config['SampleProcessing']['SMOTE_ratio'],
                           kind='svm',
                           svm_estimator=SVC(kernel='poly'),
                           n_jobs=N_jobs)

                X_train, Y_train = sm.fit_sample(X_train, Y_train)



            # Find best hyperparameters and construct svm
            svm = po.random_search_parameters(X_train, Y_train, random_state, N_jobs,
                                                 **config['HyperOptimization'])

            temp_save_data = (svm, X_train, X_test, Y_train, Y_test,
                              patient_ID_train, patient_ID_test, random_state, scaler)

            save_data.append(temp_save_data)

        [svms, X_train_set, X_test_set, Y_train_set, Y_test_set,
         patient_ID_train_set, patient_ID_test_set, seed_set, scalers] =\
            zip(*save_data)

        panda_labels = ['svms', 'X_train', 'X_test', 'Y_train', 'Y_test',
                        'config', 'patient_ID_train', 'patient_ID_test',
                        'random_seed', 'scaler']

        panda_data_temp =\
            pd.Series([svms, X_train_set, X_test_set, Y_train_set,
                       Y_test_set, config, patient_ID_train_set,
                       patient_ID_test_set, seed_set, scalers],
                      index=panda_labels,
                      name='Constructed crossvalidation')

        i_name = ''.join(i_name)
        svm_mutations[i_name] = panda_data_temp

    panda_data = pd.DataFrame(svm_mutations)

    return panda_data


def construct_all_svm(config, patient_IDs, image_features,
                      mutation_label, mutation_name):
    """
    Constructs a svm using all all patients as training set

    Arguments:
        config (Dict): Dictionary with config settings
        patient_IDs (list): IDs of the patients, used to keep track of test and
                     training sets, and genetic data
        image_features (numpy array): The values for the different features
        mutation_label (list): List of lists, where each list contains the
                                mutations status for that patient for each
                                mutations
        mutation_name (list): Contains the different mutations that are stored
                              in the mutation_label


    Returns:
        SVM_data (pandas dataframe)
    """
    svm_mutations = dict()

    N_jobs = config['General']['N_jobs']

    print('features')
    print(image_features.shape)
    for i_mutation, i_name in zip(mutation_label, mutation_name):
        i_mutation = i_mutation.ravel()
        i_mutation = i_mutation.astype(np.int)

        random_seed = np.random.randint(1, 5000)
        random_state = check_random_state(random_seed)

        print('mutation')
        print(i_mutation)

        save_data = list()

        X_train = image_features
        Y_train = i_mutation

        if config['FeatureScaling']['scale_features']:
            if config['FeatureScaling']['scaling_method'] == 'z_score':
                scaler = StandardScaler().fit(X_train)
            elif config['FeatureScaling']['scaling_method'] == 'minmax':
                scaler = MinMaxScaler().fit(X_train)
            X_train = scaler.transform(X_train)

        if config['SampleProcessing']['SMOTE']:
            sm = SMOTE(random_state=random_state,
                       ratio=config['SampleProcessing']['SMOTE_ratio'],
                       kind='svm',
                       svm_estimator=SVC(kernel='poly'),
                       n_jobs=N_jobs)

            X_train, Y_train = sm.fit_sample(X_train, Y_train)
        #
        #
        # Find best parameters to construct svm
        svm = po.random_search_parameters(X_train, Y_train, random_state, N_jobs,
                                    **config['HyperOptimization'])

        temp_save_data = (svm, X_train, Y_train, patient_IDs)

        save_data.append(temp_save_data)

        [svms, X_train_set, Y_train_set, patient_ID] =\
            zip(*save_data)

        panda_labels = ['svms', 'X_train', 'Y_train', 'patient_ID',
                        'config', 'scaler']

        panda_data_temp =\
            pd.Series([svms, X_train_set, Y_train_set, patient_ID,
                       config, scaler],
                      index=panda_labels,
                      name='Constructed crossvalidation')

        i_name = ''.join(i_name)
        svm_mutations[i_name] = panda_data_temp

    panda_data = pd.DataFrame(svm_mutations)

    return panda_data
