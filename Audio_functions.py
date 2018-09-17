import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf


def read_audio_py(py_path, sample_rate):
    #mono, _ = librosa.load(py_path, sr=sample_rate, mono=True)
    mono, native_sr = sf.read(py_path)
    if native_sr != sample_rate:
        mono = librosa.core.resample(mono, native_sr, sample_rate)
    return np.expand_dims(mono, 1).astype(np.float32)


def read_audio(path, sample_rate, n_channels=1):

    return tf.py_func(read_audio_py, [path, sample_rate], tf.float32, stateful=False)


def read_audio_pair(path_a, path_b, sample_rate):
    """
    Takes in the path of two audio files and the required output sample rate,
    returns a tuple of tensors of the wave form of the audio files.
    """
    return (tf.py_func(read_audio_py, [path_a, sample_rate], tf.float32, stateful=False),
            tf.py_func(read_audio_py, [path_b, sample_rate], tf.float32, stateful=False))


def compute_spectrogram(audio, n_fft, fft_hop, normalise=False):
    '''
    Parameters
    ----------
    audio : single to dual channel audio shaped (n_samples, )

    Returns
    -------
    Tensor of shape (n_frames, 1 + n_fft / 2, 2), where the last dimension is (magnitude, phase)
    '''

    def stft(x, normalise):
        spec = librosa.stft(
            x, n_fft=n_fft, hop_length=fft_hop, window='hann')
        mag = np.abs(spec)
        if normalise:
            # TODO: normalize?
            mag = (mag - mag.min()) / (mag.max() - mag.min())
        return mag, np.angle(spec)

    def mono_func(py_audio, normalise):
        mag, phase = stft(py_audio[:, 0], normalise)
        ret = np.array([mag, phase]).T
        return ret.astype(np.float32)

    with tf.name_scope('read_spectrogram'):
        ret = tf.py_func(mono_func, [audio, normalise], tf.float32, stateful=False)
        ret.set_shape([None, 1 + n_fft / 2, 2])
    return ret


def extract_spectrogram_patches(
        spec, n_fft, patch_window, patch_hop):
    '''
    Parameters
    ----------
    spec : Spectrogram of shape (n_frames, 1 + n_fft / 2, 2)

    Returns
    -------
    Tensor of shape (n_patches, patch_window, 1 + n_fft / 2, 2)
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


def extract_audio_patches(audio, fft_hop, patch_window, patch_hop):
    '''
    Parameters
    ----------
    audio : Waveform audio of shape (n_samples, )

    Returns
    -------
    Tensor of shape (n_patches, patch_window) containing patches from audio.
    '''
    with tf.name_scope('extract_audio_patches'):
        audio4d = tf.expand_dims(tf.expand_dims(audio, 0), 0)
        patch_length = (patch_window - 1) * fft_hop
        patch_hop_length = (patch_hop - 1) * fft_hop

        patches = tf.extract_image_patches(
            audio4d, ksizes=[1, 1, patch_length, 1],
            strides=[1, 1, patch_hop_length, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        num_patches = tf.shape(patches)[2]

        return tf.squeeze(tf.reshape(patches, [num_patches, 1, patch_length, 1]), 1)


def compute_spectrogram_map(audio_a, audio_b, n_fft, fft_hop, normalise=False):
    spec_a = compute_spectrogram(audio_a, n_fft, fft_hop, normalise)
    spec_b = compute_spectrogram(audio_b, n_fft, fft_hop, normalise)

    return spec_a, spec_b, audio_a, audio_b


def extract_patches_map(spec_a, spec_b, audio_a, audio_b, n_fft, fft_hop, patch_window, patch_hop):
    patches_a = extract_spectrogram_patches(spec_a, n_fft, patch_window, patch_hop)
    patches_b = extract_spectrogram_patches(spec_b, n_fft, patch_window, patch_hop)

    audio_patches_a = extract_audio_patches(audio_a, fft_hop, patch_window, patch_hop)
    audio_patches_b = extract_audio_patches(audio_b, fft_hop, patch_window, patch_hop)

    return patches_a, patches_b, audio_patches_a, audio_patches_b


def hwr_tf(x):
    return x * tf.cast(x > 0.0, tf.float32)


def compute_acapella_diff(mixed, noise):
    mixed_mag = mixed[:, :, 0:, :2]
    mixed_phase = mixed[:, :, 0:, 2:]
    noise_mag = noise[:, :, 0:, :2]
    voice_mag = hwr_tf(mixed_mag - noise_mag) # TODO: normalize?
    voice_phase = mixed_phase
    return mixed, noise, tf.concat((voice_mag, voice_phase), axis=3)


def spectrogramToAudioFile(magnitude, fftWindowSize, hopSize, phaseIterations=10, phase=None, length=None):
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
            reconstruction = librosa.stft(audio, fftWindowSize, hopSize)[:reconstruction.shape[0],:reconstruction.shape[1]] # Indexing to keep the output the same size
        spectrum = magnitude * np.exp(1j * np.angle(reconstruction))
        if i == phaseIterations - 1:
            audio = librosa.istft(spectrum, hopSize, length=length)
        else:
            audio = librosa.istft(spectrum, hopSize)
    return audio
