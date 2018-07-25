import tensorflow as tf
import numpy as np
import librosa


def read_audio(path, sample_rate, n_channels=1):

    def read_audio_py(py_path):
        # if n_channels == 1:
        mono, _ = librosa.load(py_path, sr=sample_rate, mono=True)
        return np.expand_dims(mono, 1)
        # elif n_channels == 2:
        # stereo, _ = librosa.load(py_path, sr=sample_rate, mono=False)
        # return stereo.T
        # else:
        # raise ValueError('Invalid channels: %d' % n_channels)

    return tf.py_func(read_audio_py, [path], tf.float32, stateful=False)


def fake_stereo(audio):

    def fake_stereo(x):
        return tf.stack([x, x], 1)

    voice = audio[:, 0]
    mixed = voice * 2
    return fake_stereo(mixed), fake_stereo(voice)


def compute_spectrogram(audio, n_fft, fft_hop, n_channels=1):
    '''
    Parameters
    ----------
    audio : single to dual channel audio shaped (n_samples, n_channels)

    Returns
    -------
    Tensor of shape (n_frames, 1 + n_fft / 2, n_channels * 2), where the
        last dimension is (left_mag, right_mag, left_phase, right_phase)
    '''

    def stft(x):
        spec = librosa.stft(
            x, n_fft=n_fft, hop_length=fft_hop, window='hann')
        # TODO: normalize?
        #mag = np.abs(spec)
        #temp = mag - mag.min()
        #mag_norm = temp / temp.max() # Return mag_norm for normalised spec.
        return np.abs(spec), np.angle(spec)

    # def stereo_func(py_audio):
    #    left_mag, left_phase = stft(py_audio[:, 0])
    #    right_mag, right_phase = stft(py_audio[:, 1])
    #    ret = np.array([left_mag, right_mag, left_phase, right_phase]).T
    #    return ret.astype(np.float32)

    def mono_func(py_audio):
        mag, phase = stft(py_audio[:, 0])
        ret = np.array([mag, phase]).T
        return ret.astype(np.float32)

    # if n_channels == 2:
    #    func = stereo_func
    # elif n_channels == 1:
    #    func = mono_func
    # else:
    #    raise ValueError('Invalid channels: %d' % n_channels)

    with tf.name_scope('read_spectrogram'):
        ret = tf.py_func(mono_func, [audio], tf.float32, stateful=False)
        ret.set_shape([None, 1 + n_fft / 2, 2])   # n_channels * 2])
    return ret


def extract_spectrogram_patches(
        spec, n_fft, n_channels, patch_window, patch_hop):
    '''
    Parameters
    ----------
    spec : Spectrogram of shape (n_frames, 1 + n_fft / 2, n_channels * 2)

    Returns
    -------
    Tensor of shape (n_patches, patch_window, 1 + n_fft / 2, n_channels * 2)
        containing patches from spec.
    '''
    with tf.name_scope('extract_spectrogram_patches'):
        spec4d = tf.expand_dims(spec, 0)

        patches = tf.extract_image_patches(
            spec4d, ksizes=[1, patch_window, 1 + n_fft / 2, 1],
            strides=[1, patch_hop, 1 + n_fft / 2, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        num_patches = tf.shape(patches)[1]

        return tf.reshape(patches, [num_patches, patch_window,
                                    int(1 + n_fft / 2), 2])


def replace_spectrogram_patches(patches):
    '''
    Parameters
    ----------
    Tensor of shape (n_patches, patch_window, 1 + n_fft / 2, n_channels * 2)
    containing patches from spec.

    Returns
    -------
    spec : Spectrogram of shape (n_frames, 1 + n_fft / 2, n_channels * 2)
    '''
    # Need to account for patch overlap
    # Is it possible to account for no set num patches per whole spectrogram?
    pass


def hwr_tf(x):
    return x * tf.cast(x > 0.0, tf.float32)


def compute_acapella_diff(mixed, noise):
    mixed_mag = mixed[:, :, 0:, :2]
    mixed_phase = mixed[:, :, 0:, 2:]
    noise_mag = noise[:, :, 0:, :2]
    voice_mag = hwr_tf(mixed_mag - noise_mag) # TODO: normalize?
    voice_phase = mixed_phase
    return mixed, noise, tf.concat((voice_mag, voice_phase), axis=3)


def spectrogramToAudioFile(magnitude, fftWindowSize, hopSize, phaseIterations=0, phase=None, length=None):
    '''
    Computes an audio signal from the given magnitude spectrogram, and optionally an initial phase.
    Griffin-Lim is executed to recover/refine the given the phase from the magnitude spectrogram.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param phase: If given, starts ISTFT with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return:
    '''
    if phase is not None:
        if phaseIterations > 0:
            # Refine audio given initial phase with a number of iterations
            return reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations, phase, length)
        # reconstructing the new complex matrix
        stftMatrix = magnitude * np.exp(phase * 1j) # magnitude * e^(j*phase)
        audio = librosa.istft(stftMatrix, hop_length=hopSize, length=length)
    else:
        audio = reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations)
    return audio

def reconPhase(magnitude, fftWindowSize, hopSize, phaseIterations=0, initPhase=None, length=None):
    '''
    Griffin-Lim algorithm for reconstructing the phase for a given magnitude spectrogram, optionally with a given
    intial phase.
    :param magnitude: Magnitudes to be converted to audio
    :param fftWindowSize: Size of FFT window used to create magnitudes
    :param hopSize: Hop size in frames used to create magnitudes
    :param phaseIterations: Number of Griffin-Lim iterations to recover phase
    :param initPhase: If given, starts reconstruction with this particular phase matrix
    :param length: If given, audio signal is clipped/padded to this number of frames
    :return:
    '''
    for i in range(phaseIterations):
        if i == 0:
            if initPhase is None:
                reconstruction = np.random.random_sample(magnitude.shape) + 1j * (2 * np.pi * np.random.random_sample(magnitude.shape) - np.pi)
            else:
                reconstruction = np.exp(initPhase * 1j) # e^(j*phase), so that angle => phase
        else:
            reconstruction = librosa.stft(audio, fftWindowSize, hopSize)
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == phaseIterations - 1:
            audio = librosa.istft(spectrum, hopSize, length=length)
        else:
            audio = librosa.istft(spectrum, hopSize)
    return audio
