import numpy as np
from collections import Counter

def load_mutation_status(genetic_file, mutation_type):
    """Loads the mutation data from a genetic file

    Args:
        genetic_file (string): The path to the genetic file
        mutation_type (list): List of the genetic mutations to load

    Returns:
        dict: A dict containing 'patient_IDs', 'mutation_label' and
         'mutation_type'
    """

    mutation_names, patient_IDs, mutation_status = load_genetic_file(
        genetic_file)

    print(mutation_type)
    mutation_label = list()
    for i_mutation in mutation_type:
        if len(i_mutation) == 1:
            mutation_index = np.where(mutation_names == i_mutation[0])[0]
            print(i_mutation[0])
            if mutation_index.size == 0:
                raise ValueError('Could not find mutation: ' + i_mutation)
            else:
                mutation_label.append(mutation_status[:, mutation_index])
        else:
            # This is a combined mutation
            mutation_index = list()
            for i_combined_mutation in i_mutation:
                mutation_index.append(
                    np.where(mutation_names == i_combined_mutation)[0])
            mutation_index = np.asarray(mutation_index)

            mutation_label.append(np.prod(mutation_status[:, mutation_index],
                                          axis=1))

    mutation_data = dict()
    mutation_data['patient_IDs'] = patient_IDs
    mutation_data['mutation_label'] = mutation_label
    mutation_data['mutation_name'] = mutation_type

    return mutation_data


def load_genetic_file(input_file):
    """
    Load the patient IDs and genetic data from the genetic file

    Args:
        input_file (string): Path of the genetic file

    Returns:
        mutation_names (numpy array): Names of the different genetic mutations
        patient_ID (numpy array): IDs of patients for which genetic data is
         loaded
        mutation_status (numpy array): The status of the different mutations
         for each patient
    """

    data = np.loadtxt(input_file, np.str)

    # Load and check the header
    header = data[0, :]
    if header[0] != 'Patient':
        raise AssertionError('First column should be patient ID!')
    else:
        # cut out the first header, only keep genetic header
        mutation_names = header[1::]

    # Patient IDs are stored in the first column
    patient_ID = data[1:, 0]

    # Mutation status is stored in all remaining columns
    mutation_status = data[1:, 1:]
    # mutation_status = mutation_status.astype(np.int)
    # Make sure no double patients
    unique_list = [item for item, count in Counter(patient_ID).iteritems() if count > 1]
    if len(unique_list) > 0:
        raise ValueError('Double patients are in list: ' + str(unique_list))

    return mutation_names, patient_ID, mutation_status
