from functools import partial
import tensorflow as tf
import Audio_functions as af
import Utils
import re


def get_dataset(
        data_folder,
        sample_rate,
        n_fft,
        fft_hop,
        n_channels,
        patch_window,
        patch_hop,
        n_parallel_readers,
        normalise):
    # TODO Still need to fix this to stop it producing a tuple
    return (
        tf.data.Dataset.list_files(data_folder + '/*.wav')  # TODO still uncertain if this is done in deterministic order or not
        .filter(lambda x: re.search('CH0', str(x)) is None)  # Filter out any files containing 'CH0' as these do not exist in the mixed data
        .map(partial(af.read_audio,
                     sample_rate=sample_rate,
                     n_channels=n_channels),
             num_parallel_calls=n_parallel_readers)
        .map(Utils.partial_argv(af.compute_spectrogram,
                                n_fft=n_fft,
                                fft_hop=fft_hop,
                                n_channels=n_channels,
                                normalise=normalise),
             num_parallel_calls=n_parallel_readers)
        .map(Utils.partial_argv(af.extract_spectrogram_patches,
                                n_fft=n_fft,
                                n_channels=n_channels,
                                patch_window=patch_window,
                                patch_hop=patch_hop,))
        .flat_map(Utils.zip_tensor_slices))


def zip_datasets(dataset_a, dataset_b, n_shuffle, batch_size, shuffle):
    if shuffle:
        return tf.data.Dataset.zip((dataset_a, dataset_b))\
            .batch(batch_size)\
            .shuffle(n_shuffle)
    else:
        return tf.data.Dataset.zip((dataset_a, dataset_b)) \
            .batch(batch_size)


def prepare_datasets(model_config):

    def build_datasets(model_config, path):
        x_train = get_dataset(model_config['data_root'] + path['x_train'],
                              model_config['SAMPLE_RATE'],
                              model_config['N_FFT'],
                              model_config['FFT_HOP'],
                              model_config['N_CHANNELS'],
                              model_config['PATCH_WINDOW'],
                              model_config['PATCH_HOP'],
                              model_config['N_PARALLEL_READERS'],
                              model_config['NORMALISE_MAG'])
        y_train = get_dataset(model_config['data_root'] + path['y_train'],
                              model_config['SAMPLE_RATE'],
                              model_config['N_FFT'],
                              model_config['FFT_HOP'],
                              model_config['N_CHANNELS'],
                              model_config['PATCH_WINDOW'],
                              model_config['PATCH_HOP'],
                              model_config['N_PARALLEL_READERS'],
                              model_config['NORMALISE_MAG'])
        train = zip_datasets(x_train, y_train, model_config['N_SHUFFLE'], model_config['BATCH_SIZE'], shuffle=True)

        x_val = get_dataset(model_config['data_root'] + path['x_val'],
                            model_config['SAMPLE_RATE'],
                            model_config['N_FFT'],
                            model_config['FFT_HOP'],
                            model_config['N_CHANNELS'],
                            model_config['PATCH_WINDOW'],
                            model_config['PATCH_HOP'],
                            model_config['N_PARALLEL_READERS'],
                            model_config['NORMALISE_MAG'])
        y_val = get_dataset(model_config['data_root'] + path['y_val'],
                            model_config['SAMPLE_RATE'],
                            model_config['N_FFT'],
                            model_config['FFT_HOP'],
                            model_config['N_CHANNELS'],
                            model_config['PATCH_WINDOW'],
                            model_config['PATCH_HOP'],
                            model_config['N_PARALLEL_READERS'],
                            model_config['NORMALISE_MAG'])
        val = zip_datasets(x_val, y_val, model_config['N_SHUFFLE'], model_config['BATCH_SIZE'], shuffle=False)

        x_test = get_dataset(model_config['data_root'] + path['x_test'],
                             model_config['SAMPLE_RATE'],
                             model_config['N_FFT'],
                             model_config['FFT_HOP'],
                             model_config['N_CHANNELS'],
                             model_config['PATCH_WINDOW'],
                             model_config['PATCH_HOP'],
                             model_config['N_PARALLEL_READERS'],
                             model_config['NORMALISE_MAG'])
        y_test = get_dataset(model_config['data_root'] + path['y_test'],
                             model_config['SAMPLE_RATE'],
                             model_config['N_FFT'],
                             model_config['FFT_HOP'],
                             model_config['N_CHANNELS'],
                             model_config['PATCH_WINDOW'],
                             model_config['PATCH_HOP'],
                             model_config['N_PARALLEL_READERS'],
                             model_config['NORMALISE_MAG'])
        test = zip_datasets(x_test, y_test, model_config['N_SHUFFLE'], model_config['BATCH_SIZE'], shuffle=False)

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
                    'y_train': 'tr05_bth',
                    'x_val': 'dt05_' + string,
                    'y_val': 'dt05_bth',
                    'x_test': 'et05_' + string,
                    'y_test': 'et05_bth'}
            sets.append(build_datasets(model_config, path))
        train_data = sets[0][0].concatenate(sets[1][0].concatenate(sets[2][0].concatenate(sets[3][0])))
        val_data = sets[0][1].concatenate(sets[1][1].concatenate(sets[2][1].concatenate(sets[3][1])))
        test_data = sets[0][2].concatenate(sets[1][2].concatenate(sets[2][2].concatenate(sets[3][2])))

    return train_data, val_data, test_data
