import sys
import os
import csv
import pickle
import datetime
from glob import glob
import numpy as np
import mir_eval
import audio_functions as af

"""
This script takes the results of a test set being passed through a model, converts the relevant parts from spectrogram 
to audio and then calculates audio quality metrics.

This process causes a bottleneck and is not on the critical path towards training a model. As such, required data has 
been dumped to pickle files by the main script, so that this script can be run on a different server.
"""


def get_test_metrics(argv):

    experiment_id = argv[1]
    if len(argv) == 3:
        phase_iterations = int(argv[2])
    else:
        phase_iterations = 0

    # Calculate number of test runs in experiment
    dump_folder = 'dumps/' + experiment_id
    file_list = glob(dump_folder + '/*')
    test_num = max([int(file.split('_')[2]) for file in file_list]) + 1
    metrics = list()
    #  For each test run, calculate the results
    for test in range(test_num):
        print('{ts}:\tProcessing test {t}'.format(ts=datetime.datetime.now(), t=test))
        test_files = [file for file in file_list if file.split('_')[2] == str(test)]
        test_costs = list()
        sdrs = np.empty((0, 2))
        sirs = np.empty((0, 2))
        sars = np.empty((0, 2))
        nsdrs = np.empty((0, 2))

        #  There will be one pickle file per batch. For each one, load it and calculate the metrics
        for file in test_files:
            cost, voice_est_mag, voice_ref_mag, voice_ref_audio, \
                mixed_audio, mixed_mag, mixed_phase, model_config = pickle.load(open(file, 'rb'))
            print('{ts}:\t{f} loaded.'.format(ts=datetime.datetime.now(), f=file))
            test_costs.append(cost)
            background_ref_mag = mixed_mag - voice_ref_mag
            background_est_mag = mixed_mag - voice_est_mag
            for i in range(voice_est_mag.shape[0]):
                # Transform output back to audio
                #print('{ts}:\treconstructing audio {i}.'.format(ts=datetime.datetime.now(), i=i))
                voice_est_audio = af.spectrogramToAudioFile(np.squeeze(voice_est_mag[i, :, :, :]).T,
                                                            model_config['n_fft'], model_config['fft_hop'],
                                                            phaseIterations=phase_iterations,
                                                            phase=np.squeeze(mixed_phase[i, :, :, :]).T)
                background_ref_audio = af.spectrogramToAudioFile(np.squeeze(background_ref_mag[i, :, :, :]).T,
                                                                 model_config['n_fft'], model_config['fft_hop'],
                                                                 phaseIterations=phase_iterations,
                                                                 phase=np.squeeze(mixed_phase[i, :, :, :]).T)
                background_est_audio = af.spectrogramToAudioFile(np.squeeze(background_est_mag[i, :, :, :]).T,
                                                                 model_config['n_fft'],  model_config['fft_hop'],
                                                                 phaseIterations=phase_iterations,
                                                                 phase=np.squeeze(mixed_phase[i, :, :, :]).T)
                #print('{ts}:\taudio reconstructed{i}.'.format(ts=datetime.datetime.now(), i=i))
                # Reshape for mir_eval
                voice_est_audio = np.expand_dims(voice_est_audio, 1).T
                background_ref_audio = np.expand_dims(background_ref_audio, 1).T
                background_est_audio = np.expand_dims(background_est_audio, 1).T
                voice_ref_audio_patch = voice_ref_audio[i, :, :].T
                mixed_audio_patch = mixed_audio[i, :, :].T

                ref_sources = np.concatenate((voice_ref_audio_patch, background_ref_audio), axis=0)
                est_sources = np.concatenate((voice_est_audio, background_est_audio), axis=0)
                mixed_sources = np.concatenate((mixed_audio_patch, mixed_audio_patch), axis=0)

                # Calculate audio quality statistics
                #print('{ts}:\tcalculating metrics {i}.'.format(ts=datetime.datetime.now(), i=i))
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(ref_sources, est_sources, compute_permutation=False)
                sdr_mr, _, _, _ = mir_eval.separation.bss_eval_sources(ref_sources, mixed_sources, compute_permutation=False)
                nsdr = sdr - sdr_mr
                #print('{ts}:\tmetrics calculated {i}.'.format(ts=datetime.datetime.now(), i=i))

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

    return metrics


#test_metrics = get_test_metrics(['test', '50'])
test_metrics = get_test_metrics(sys.argv)
print('{ts}:\nProcessing complete\n{t}'.format(ts=datetime.datetime.now(), t=test_metrics))

