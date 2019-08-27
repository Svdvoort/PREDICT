import IO.config_io as config_io
import get_image_features
import Genetics.genetic_processing as gp
import classifiers.interface

config_path = '/home/svandervoort/Repos/PREDICT/Settings/Mixed.ini'

for ii in range(1, 2):


    # Load variables from the confilg file
    config = config_io.load_config(config_path)
    save_name = config['DataPaths']['svm_file'] + str(ii) + '.hdf'

    mutation_data = gp.load_mutation_status(**config['Genetics'])

    # Calculate the image features
    print('Calculating features')
    image_features = get_image_features.get_image_features(patient_IDs=mutation_data['patient_IDs'],
                                                           **config['ImageFeatures'])


    # Get the mutation labels and patient IDs
    # print(mutation_data['mutation_label'])

    print('Total of ' + str(mutation_data['patient_IDs'].shape[0]) + ' patients')
    print('Image features:' + str(image_features.shape))

    print('Constructing classifier')
    classifier = classifiers.interface.construct_classifier(config, mutation_data,
                                                            image_features)


    print("Saving data!")
    classifier.to_hdf(save_name, 'SVMdata')
    print("Done, enjoy your fresh, oven-baked prediction!")
