from functools import partial
import tensorflow as tf
import Audio_functions as af
import Utils


def get_dataset(
        data_folder,
        sample_rate,
        n_fft,
        fft_hop,
        n_channels,
        patch_window,
        patch_hop,
        n_parallel_readers
):
    """Still need to fix this to stop it producing a tuple"""
    return (
        tf.data.Dataset.list_files(data_folder + '/*.wav')
        .map(partial(
            af.read_audio,
            sample_rate=sample_rate,
            n_channels=n_channels
        ), num_parallel_calls=n_parallel_readers)
        #.map(fake_stereo, num_parallel_calls=n_parallel_readers)
        .map(Utils.partial_argv(
            af.compute_spectrogram,
            n_fft=n_fft,
            fft_hop=fft_hop,
            n_channels=n_channels,
        ), num_parallel_calls=n_parallel_readers)
        .map(Utils.partial_argv(
            af.extract_spectrogram_patches,
            n_fft=n_fft,
            n_channels=n_channels,
            patch_window=patch_window,
            patch_hop=patch_hop,
        ))
        .flat_map(Utils.zip_tensor_slices)
    )


def zip_datasets(dataset_a, dataset_b, n_shuffle, batch_size):
    return tf.data.Dataset.zip((dataset_a, dataset_b))\
        .batch(batch_size)\
        .shuffle(n_shuffle)


def prepare_datasets(model_config):

    x_train = get_dataset(model_config['data_root']+'train_sup/Mixed',
                                  model_config['SAMPLE_RATE'],
                                  model_config['N_FFT'],
                                  model_config['FFT_HOP'],
                                  model_config['N_CHANNELS'],
                                  model_config['PATCH_WINDOW'],
                                  model_config['PATCH_HOP'],
                                  model_config['N_PARALLEL_READERS'])
    y_train = get_dataset(model_config['data_root']+'train_sup/Voice',
                                  model_config['SAMPLE_RATE'],
                                  model_config['N_FFT'],
                                  model_config['FFT_HOP'],
                                  model_config['N_CHANNELS'],
                                  model_config['PATCH_WINDOW'],
                                  model_config['PATCH_HOP'],
                                  model_config['N_PARALLEL_READERS'])
    train_data = zip_datasets(x_train, y_train, model_config['N_SHUFFLE'], model_config['BATCH_SIZE'])

    x_val = get_dataset(model_config['data_root']+'validation/Mixed',
                                model_config['SAMPLE_RATE'],
                                model_config['N_FFT'],
                                model_config['FFT_HOP'],
                                model_config['N_CHANNELS'],
                                model_config['PATCH_WINDOW'],
                                model_config['PATCH_HOP'],
                                model_config['N_PARALLEL_READERS'])
    y_val = get_dataset(model_config['data_root']+'validation/Voice',
                                model_config['SAMPLE_RATE'],
                                model_config['N_FFT'],
                                model_config['FFT_HOP'],
                                model_config['N_CHANNELS'],
                                model_config['PATCH_WINDOW'],
                                model_config['PATCH_HOP'],
                                model_config['N_PARALLEL_READERS'])
    val_data = zip_datasets(x_val, y_val, model_config['N_SHUFFLE'], model_config['BATCH_SIZE'])

    x_test = get_dataset(model_config['data_root']+'test/Mixed',
                                 model_config['SAMPLE_RATE'],
                                 model_config['N_FFT'],
                                 model_config['FFT_HOP'],
                                 model_config['N_CHANNELS'],
                                 model_config['PATCH_WINDOW'],
                                 model_config['PATCH_HOP'],
                                 model_config['N_PARALLEL_READERS'])
    y_test = get_dataset(model_config['data_root']+'test/Voice',
                                 model_config['SAMPLE_RATE'],
                                 model_config['N_FFT'],
                                 model_config['FFT_HOP'],
                                 model_config['N_CHANNELS'],
                                 model_config['PATCH_WINDOW'],
                                 model_config['PATCH_HOP'],
                                 model_config['N_PARALLEL_READERS'])
    test_data = zip_datasets(x_test, y_test, model_config['N_SHUFFLE'], model_config['BATCH_SIZE'])

    return train_data, val_data, test_data
