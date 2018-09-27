from functools import partial
import numpy as np
import tensorflow as tf
import audio_functions as af
import re
import os
from glob import glob


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
            if 'CHiME' in directory_a:
                if file_a[:13] == file_b[:13] and (file_a[17:] == file_b[17:] or len(file_a) != len(file_b)):
                    zipped_list.append((str(directory_a + '/' + file_a), str(directory_b + '/' + file_b)))
                    if len(file_a) == len(file_b):
                        filelist_b.remove(file_b)
                        break
            else:
                if file_a == file_b:
                    zipped_list.append((str(directory_a + file_a), str(directory_b + file_b)))
                    filelist_b.remove(file_b)
                    break

    if len(zipped_list) == 0:
        zipped_list = np.empty((0, 2))
    else:
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
                       normalise,
                       mag_phase):

    return (
        tf.data.Dataset.from_tensor_slices((zipped_files[:, 0], zipped_files[:, 1]))
        .map(partial(af.read_audio_pair,
                     sample_rate=sample_rate),
             num_parallel_calls=n_parallel_readers)
        .map(partial(af.extract_audio_patches_map,
                     fft_hop=fft_hop,
                     patch_window=patch_window,
                     patch_hop=patch_hop,),
             num_parallel_calls=n_parallel_readers)
        .flat_map(af.zip_tensor_slices)
        .map(partial(af.compute_spectrogram_map,
                     n_fft=n_fft,
                     fft_hop=fft_hop,
                     normalise=normalise,
                     mag_phase=mag_phase),
             num_parallel_calls=n_parallel_readers)
        .shuffle(n_shuffle).batch(batch_size).prefetch(3)
    )

def prepare_datasets(model_config):

    def build_datasets(model_config, root, path):
        train_files = zip_files(os.path.join(root, path['x_train']),
                                os.path.join(root, path['y_train']))
        train = get_paired_dataset(train_files,
                                   model_config['sample_rate'],
                                   model_config['n_fft'],
                                   model_config['fft_hop'],
                                   model_config['patch_window'],
                                   model_config['patch_hop'],
                                   model_config['n_parallel_readers'],
                                   model_config['batch_size'],
                                   model_config['n_shuffle'],
                                   model_config['normalise_mag'],
                                   model_config['mag_phase'])

        val_files = zip_files(os.path.join(root, path['x_val']),
                              os.path.join(root, path['y_val']))
        val = get_paired_dataset(val_files,
                                 model_config['sample_rate'],
                                 model_config['n_fft'],
                                 model_config['fft_hop'],
                                 model_config['patch_window'],
                                 model_config['patch_hop'],
                                 model_config['n_parallel_readers'],
                                 model_config['batch_size'],
                                 model_config['n_shuffle'],
                                 model_config['normalise_mag'],
                                 model_config['mag_phase'])

        test_files = zip_files(os.path.join(root, path['x_test']),
                               os.path.join(root, path['y_test']))
        test = get_paired_dataset(test_files,
                                  model_config['sample_rate'],
                                  model_config['n_fft'],
                                  model_config['fft_hop'],
                                  model_config['patch_window'],
                                  model_config['patch_hop'],
                                  model_config['n_parallel_readers'],
                                  model_config['batch_size'],
                                  model_config['n_shuffle'],
                                  model_config['normalise_mag'],
                                  model_config['mag_phase'])
        return train, val, test

    if model_config['local_run']:  # If running on local machine, mini dataset is all in one folder
        path = {'x_train': 'train_sup/Mixed',
                'y_train': 'train_sup/Voice',
                'x_val': 'validation/Mixed',
                'y_val': 'validation/Voice',
                'x_test': 'test/Mixed',
                'y_test': 'test/Voice'}
        train_data, val_data, test_data = build_datasets(model_config, model_config['data_root'], path)
        return train_data, val_data, test_data

    else:  # If running on server, data is in several folders and requires concatenation
        if model_config['dataset'] in ['both', 'CHiME']:
            # Get CHiME data
            sets = list()
            for string in ['bus_simu/', 'caf_simu/', 'ped_simu/', 'str_simu/']:
                path = {'x_train': 'tr05_' + string,
                        'y_train': 'tr05_org',
                        'x_val': 'dt05_' + string,
                        'y_val': 'dt05_bth',
                        'x_test': 'et05_' + string,
                        'y_test': 'et05_bth'}
                sets.append(build_datasets(model_config, model_config['chime_data_root'], path))
            chime_train_data = sets[0][0].concatenate(sets[1][0].concatenate(sets[2][0].concatenate(sets[3][0])))
            chime_val_data = sets[0][1].concatenate(sets[1][1].concatenate(sets[2][1].concatenate(sets[3][1])))
            chime_test_data = sets[0][2].concatenate(sets[1][2].concatenate(sets[2][2].concatenate(sets[3][2])))

        if model_config['dataset'] in ['both', 'LibriSpeech']:
            # Get list of LibriSpeech sub-directories
            voice_train_dirs = glob(model_config['librispeech_data_root'] + 'Voice/train-clean-100/**/', recursive=True)
            voice_train_dirs.extend(
                glob(model_config['librispeech_data_root'] + 'Voice/train-clean-360/**/', recursive=True))
            voice_train_dirs.extend(
                glob(model_config['librispeech_data_root'] + 'Voice/train-other-500/**/', recursive=True))
            voice_val_dirs = glob(model_config['librispeech_data_root'] + 'Voice/dev-clean/**/', recursive=True)
            voice_test_dirs = glob(model_config['librispeech_data_root'] + 'Voice/test-clean/**/', recursive=True)

            mix_train_dirs = glob(model_config['librispeech_data_root'] + 'Mixed/train-clean-100/**/', recursive=True)
            mix_train_dirs.extend(
                glob(model_config['librispeech_data_root'] + 'Mixed/train-clean-360/**/', recursive=True))
            mix_train_dirs.extend(
                glob(model_config['librispeech_data_root'] + 'Mixed/train-other-500/**/', recursive=True))
            mix_val_dirs = glob(model_config['librispeech_data_root'] + 'Mixed/dev-clean/**/', recursive=True)
            mix_test_dirs = glob(model_config['librispeech_data_root'] + 'Mixed/test-clean/**/', recursive=True)

            # Check corresponding list are of equal length
            assert len(voice_train_dirs) == len(mix_train_dirs)
            assert len(voice_val_dirs) == len(mix_val_dirs)
            assert len(voice_test_dirs) == len(mix_test_dirs)

            train_file_list = np.empty((0, 2))
            for i in range(len(voice_train_dirs)):
                train_file_list = np.concatenate((train_file_list, zip_files(mix_train_dirs[i], voice_train_dirs[i])), axis=0)
            libri_train_data = get_paired_dataset(train_file_list,
                                                  model_config['sample_rate'],
                                                  model_config['n_fft'],
                                                  model_config['fft_hop'],
                                                  model_config['patch_window'],
                                                  model_config['patch_hop'],
                                                  model_config['n_parallel_readers'],
                                                  model_config['batch_size'],
                                                  model_config['n_shuffle'],
                                                  model_config['normalise_mag'],
                                                  model_config['mag_phase'])

            val_file_list = np.empty((0, 2))
            for i in range(len(voice_val_dirs)):
                val_file_list = np.concatenate((val_file_list, zip_files(mix_val_dirs[i], voice_val_dirs[i])), axis=0)
            libri_val_data = get_paired_dataset(val_file_list,
                                                model_config['sample_rate'],
                                                model_config['n_fft'],
                                                model_config['fft_hop'],
                                                model_config['patch_window'],
                                                model_config['patch_hop'],
                                                model_config['n_parallel_readers'],
                                                model_config['batch_size'],
                                                model_config['n_shuffle'],
                                                model_config['normalise_mag'],
                                                model_config['mag_phase'])

            test_file_list = np.empty((0, 2))
            for i in range(len(voice_test_dirs)):
                test_file_list = np.concatenate((test_file_list, zip_files(mix_test_dirs[i], voice_test_dirs[i])), axis=0)
            libri_test_data = get_paired_dataset(test_file_list,
                                                 model_config['sample_rate'],
                                                 model_config['n_fft'],
                                                 model_config['fft_hop'],
                                                 model_config['patch_window'],
                                                 model_config['patch_hop'],
                                                 model_config['n_parallel_readers'],
                                                 model_config['batch_size'],
                                                 model_config['n_shuffle'],
                                                 model_config['normalise_mag'],
                                                 model_config['mag_phase'])

        if model_config['dataset'] == 'CHiME':
            return chime_train_data, chime_val_data, chime_test_data
        elif model_config['dataset'] == 'LibriSpeech':
            return libri_train_data, libri_val_data, libri_test_data
        elif model_config['dataset'] == 'both':
            return chime_train_data.concatenate(libri_train_data), \
                   chime_val_data.concatenate(libri_val_data), \
                   chime_test_data.concatenate(libri_test_data)
