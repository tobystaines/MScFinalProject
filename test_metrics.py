import sys
import os
import csv
import pickle
import shutil
import datetime
from glob import glob
import numpy as np
import mir_eval
import librosa
import audio_functions as af

"""
This script takes the results of a test set being passed through a model, converts the relevant parts from spectrogram 
to audio and then calculates audio quality metrics.

This process causes a bottleneck and is not on the critical path towards training a model. As such, required data has 
been dumped to pickle files by the main script, so that this script can be run on a different server or post training.
"""


def get_test_metrics(argv):
    """
    Calculate the audio separation quality metrics from pickled test results.
    """

    experiment_id = argv[1]

    if argv[2] == 'mag_phase':
        mag_phase = True
    elif argv[2] == 'complex':
        mag_phase = False
    else:
        raise Exception("User must specify 'mag_phase' or 'complex' as second argument")

    if len(argv) == 4:
        phase_iterations = int(argv[3])
    else:
        phase_iterations = 0

    # Calculate number of test runs in experiment
    dump_folder = 'dumps/' + experiment_id
    file_list = glob(dump_folder + '/*')
    test_num = max([int(file.split('_')[2]) for file in file_list]) + 1
    batch_num = max([int(file.split('_')[4]) for file in file_list]) + 1
    metrics = []
    #  For each test run, calculate the results
    for test in range(test_num):
        print('{ts}:\tProcessing test {t}'.format(ts=datetime.datetime.now(), t=test))
        print('\t\t{b} batches to process.'.format(b=batch_num))
        test_files = [file for file in file_list if file.split('_')[2] == str(test)]
        test_costs = []
        sdrs = np.empty((0, 2))
        sirs = np.empty((0, 2))
        sars = np.empty((0, 2))
        nsdrs = np.empty((0, 2))

        #  There will be one pickle file per batch. For each one, load it and calculate the metrics
        for file in test_files:
            if mag_phase:  # Data to be processed is from a magnitude spectrogram based model
                cost, voice_est_mag, voice_ref_mag, voice_ref_audio, \
                    mixed_audio, mixed_mag, mixed_phase, model_config = pickle.load(open(file, 'rb'))
                print('{ts}:\t{f} loaded.'.format(ts=datetime.datetime.now(), f=file))
                test_costs.append(cost)
            else:  # Data to be processed is from a complex number spectrogram based model
                cost, voice_est_spec, voice_ref_spec, \
                    voice_ref_audio, mixed_audio, mixed_spec, model_config = pickle.load(open(file, 'rb'))
                print('{ts}:\t{f} loaded.'.format(ts=datetime.datetime.now(), f=file))
                test_costs.append(cost)
                mixed_phase = np.angle(mixed_spec)

            voice_est_audio = np.empty(voice_ref_audio.shape)
            for i in range(voice_est_audio.shape[0]):
                # Transform output back to audio
                if mag_phase:
                    wave = af.spectrogramToAudioFile(voice_est_mag[i, :, :].T,
                                                     model_config['n_fft'], model_config['fft_hop'],
                                                     phaseIterations=phase_iterations,
                                                     phase=mixed_phase[i, :, :].T)
                else:
                    wave = librosa.istft(voice_est_spec[i, :, :].T, model_config['fft_hop'])

                voice_est_audio[i, :, :] = np.expand_dims(wave, axis=1)

                # Normalise the waveforms to enable background noise calculation by subtraction
                voice_ref_audio[i, :, :] = af.normalise_audio(voice_ref_audio[i, :, :])
                voice_est_audio[i, :, :] = af.normalise_audio(voice_est_audio[i, :, :])
                mixed_audio[i, :, :] = af.normalise_audio(mixed_audio[i, :, :])

            # Reshape for mir_eval
            voice_ref_audio = np.transpose(voice_ref_audio, (0, 2, 1))
            voice_est_audio = np.transpose(voice_est_audio, (0, 2, 1))
            mixed_audio = np.transpose(mixed_audio, (0, 2, 1))

            # Subtract voice to calculate background noise
            background_ref_audio = mixed_audio - voice_ref_audio
            background_est_audio = mixed_audio - voice_est_audio

            for i in range(voice_est_audio.shape[0]):
                ref_sources = np.concatenate((voice_ref_audio[i, :, :], background_ref_audio[i, :, :]), axis=0)
                est_sources = np.concatenate((voice_est_audio[i, :, :], background_est_audio[i, :, :]), axis=0)
                mixed_sources = np.concatenate((mixed_audio[i, :, :], mixed_audio[i, :, :]), axis=0)

                # Calculate audio quality statistics
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref_sources, est_sources, compute_permutation=False)
                sdr_mr, _, _, _ = mir_eval.separation.bss_eval_sources(ref_sources, mixed_sources, compute_permutation=False)
                nsdr = sdr - sdr_mr
                sdrs = np.concatenate((sdrs, np.expand_dims(sdr, 1).T), axis=0)
                sirs = np.concatenate((sirs, np.expand_dims(sir, 1).T), axis=0)
                sars = np.concatenate((sars, np.expand_dims(sar, 1).T), axis=0)
                nsdrs = np.concatenate((nsdrs, np.expand_dims(nsdr, 1).T), axis=0)
            print('{ts}:\t{f} processed.'.format(ts=datetime.datetime.now(), f=file))

        #  Record mean results for each metric across all batches in the test
        mean_cost = sum(test_costs) / len(test_costs)
        mean_sdr = np.mean(sdrs, axis=0)
        mean_sir = np.mean(sirs, axis=0)
        mean_sar = np.mean(sars, axis=0)
        mean_nsdr = sum(nsdrs) / len(nsdrs)
        for (k, v) in (('voice', 0), ('background', 1)):
            metrics.append({'test': str(test) + '_' + k, 'mean_cost': mean_cost, 'mean_sdr': mean_sdr[v],
                            'mean_sir': mean_sir[v], 'mean_sar': mean_sar[v], 'mean_nsdr': mean_nsdr[v]})

    #  Write the results from the experiment to a CSV file, one row per test
    if not os.path.isdir('test_metrics'):
        os.mkdir('test_metrics')
    file_name = 'test_metrics/' + experiment_id + '.csv'
    with open(file_name, 'w') as csvfile:
        fieldnames = ['test', 'mean_cost', 'mean_sdr', 'mean_sir', 'mean_sar', 'mean_nsdr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        for test in metrics:
            writer.writerow(test)

    # Delete the pickle files, as they are enormous, and no longer needed
    print('Deleting pickle files')
    shutil.rmtree(dump_folder)

    return metrics


#test_metrics = get_test_metrics(['test', '50', 'mag_phase'])
test_metrics = get_test_metrics(sys.argv)
print('{ts}:\nProcessing complete\n{t}'.format(ts=datetime.datetime.now(), t=test_metrics))

