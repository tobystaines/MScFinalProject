from functools import partial
import numpy as np
import tensorflow as tf
import Audio_functions as af
import Utils
import re
import os


def zip_files(directory_a, directory_b):
    """
    Takes in two directories (a and b) and returns an array, where each row is a pair of matching file paths,
    one from each directory, with directory a in col 0 and directory b in col 1
    """

    filelist_a = [f for f in os.listdir(directory_a) if
                  os.path.isfile(os.path.join(directory_a, f)) and re.search('CH0', f) is None]
    filelist_b = [f for f in os.listdir(directory_b) if
                  os.path.isfile(os.path.join(directory_b, f)) and re.search('CH0', f) is None]

    zipped_list = list()

    for file_a in filelist_a:
        for file_b in filelist_b:
            if file_a[:13] == file_b[:13] and (file_a[17:] == file_b[17:] or len(file_a) != len(file_b)):
                zipped_list.append((str(directory_a + '/' + file_a), str(directory_b + '/' + file_b)))
                if len(file_a) == len(file_b):
                    filelist_b.remove(file_b)
                break

    zipped_list = np.array(zipped_list)

    return zipped_list


def get_paired_dataset(zipped_files,
                       sample_rate,
                       n_fft,
                       fft_hop,
                       patch_window,
                       patch_hop,
                       n_parallel_readers,
                       batch_size,
                       n_shuffle,
                       normalise):

    return (
        tf.data.Dataset.from_tensor_slices((zipped_files[:,0],zipped_files[:,1]))
        .map(partial(af.read_audio_pair,
                     sample_rate=sample_rate),
             num_parallel_calls=n_parallel_readers)
        .map(partial(af.compute_spectrogram_map,
                     n_fft=n_fft,
                     fft_hop=fft_hop,
                     normalise=normalise),
             num_parallel_calls=n_parallel_readers)
        .map(partial(af.extract_patches_map,
                     n_fft=n_fft,
                     fft_hop = fft_hop,
                     patch_window=patch_window,
                     patch_hop=patch_hop,),
             num_parallel_calls=n_parallel_readers)
        .flat_map(Utils.zip_tensor_slices).batch(batch_size).shuffle(n_shuffle))


def prepare_datasets(model_config):

    def build_datasets(model_config, path):
        train_files = zip_files(os.path.join(model_config['data_root'], path['x_train']),
                                os.path.join(model_config['data_root'], path['y_train']))
        train = get_paired_dataset(train_files,
                                   model_config['SAMPLE_RATE'],
                                   model_config['N_FFT'],
                                   model_config['FFT_HOP'],
                                   model_config['PATCH_WINDOW'],
                                   model_config['PATCH_HOP'],
                                   model_config['N_PARALLEL_READERS'],
                                   model_config['BATCH_SIZE'],
                                   model_config['N_SHUFFLE'],
                                   model_config['NORMALISE_MAG'])

        val_files = zip_files(os.path.join(model_config['data_root'], path['x_val']),
                              os.path.join(model_config['data_root'], path['y_val']))
        val = get_paired_dataset(val_files,
                                 model_config['SAMPLE_RATE'],
                                 model_config['N_FFT'],
                                 model_config['FFT_HOP'],
                                 model_config['PATCH_WINDOW'],
                                 model_config['PATCH_HOP'],
                                 model_config['N_PARALLEL_READERS'],
                                 model_config['BATCH_SIZE'],
                                 model_config['N_SHUFFLE'],
                                 model_config['NORMALISE_MAG'])

        test_files = zip_files(os.path.join(model_config['data_root'], path['x_test']),
                               os.path.join(model_config['data_root'], path['y_test']))
        test = get_paired_dataset(test_files,
                                  model_config['SAMPLE_RATE'],
                                  model_config['N_FFT'],
                                  model_config['FFT_HOP'],
                                  model_config['PATCH_WINDOW'],
                                  model_config['PATCH_HOP'],
                                  model_config['N_PARALLEL_READERS'],
                                  model_config['BATCH_SIZE'],
                                  model_config['N_SHUFFLE'],
                                  model_config['NORMALISE_MAG'])
        return train, val, test

    if model_config['local_run']:  # If running on local machine, mini dataset is all in one folder
        path = {'x_train': 'train_sup/Mixed',
                'y_train': 'train_sup/Voice',
                'x_val': 'validation/Mixed',
                'y_val': 'validation/Voice',
                'x_test': 'test/Mixed',
                'y_test': 'test/Voice'}
        train_data, val_data, test_data = build_datasets(model_config, path)
    else:  # If running on server, data is in several folders and requires concatenation
        sets = list()
        for string in ['bus_simu', 'caf_simu', 'ped_simu', 'str_simu']:
            path = {'x_train': 'tr05_' + string,
                    'y_train': 'tr05_org',
                    'x_val': 'dt05_' + string,
                    'y_val': 'dt05_bth',
                    'x_test': 'et05_' + string,
                    'y_test': 'et05_bth'}
            sets.append(build_datasets(model_config, path))
        train_data = sets[0][0].concatenate(sets[1][0].concatenate(sets[2][0].concatenate(sets[3][0])))
        val_data = sets[0][1].concatenate(sets[1][1].concatenate(sets[2][1].concatenate(sets[3][1])))
        test_data = sets[0][2].concatenate(sets[1][2].concatenate(sets[2][2].concatenate(sets[3][2])))

    return train_data, val_data, test_data
