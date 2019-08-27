import pandas as pd
import numpy as np


def get_patient_features(metadata):
    patient_age = int(metadata[0x10, 0x1010].value[0:3])
    patient_sex = metadata[0x10, 0x40].value

    if patient_sex == 'M':
        patient_sex = 0
    elif patient_sex == 'F':
        patient_sex = 1
    else:
        raise ValueError

    panda_labels = ['patient_age', 'patient_sex']
    patient_features = [patient_age, patient_sex]

    panda_dict = dict(zip(panda_labels, patient_features))
    patient_features = pd.Series(panda_dict)

    return patient_features


def get_patient_features_file(patient_ID, patient_file):
    patient_data = np.loadtxt(patient_file, np.str, delimiter='\t', skiprows=1)

    patient_location = np.argwhere(patient_ID == patient_data[:, 0])

    if len(patient_location) == 0:
        raise ValueError

    patient_location = patient_location[0][0]

    patient_age = int(patient_data[patient_location, 1])

    patient_sex = patient_data[patient_location, 2]

    if patient_sex == 'M':
        patient_sex = 0
    elif patient_sex == 'F':
        patient_sex = 1
    else:
        raise ValueError

    panda_labels = ['patient_age', 'patient_sex']
    patient_features = [patient_age, patient_sex]


    panda_dict = dict(zip(panda_labels, patient_features))
    patient_features = pd.Series(panda_dict)

    return patient_features
