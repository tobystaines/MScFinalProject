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


def get_test_metrics(experiment_id):

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
        sdrs = list()
        sirs = list()
        sars = list()
        nsdrs = list()

        #  There will be one pickle file per batch. For each one, load it and calculate the metrics
        for file in test_files:
            cost, voice_est_mag, voice, mixed_audio, mixed_phase, model_config = pickle.load(open(file, 'rb'))
            print('{ts}:\t{f} loaded.'.format(ts=datetime.datetime.now(), f=file))
            test_costs.append(cost)
            for i in range(voice_est_mag.shape[0]):
                # Transform output back to audio
                #print('{ts}:\treconstructing audio {i}.'.format(ts=datetime.datetime.now(), i=i))
                voice_est = af.spectrogramToAudioFile(np.squeeze(voice_est_mag[i, :, :, :]).T, model_config['N_FFT'],
                                                      model_config['FFT_HOP'], phase=np.squeeze(mixed_phase[i, :, :, :]).T)
                #print('{ts}:\taudio reconstructed{i}.'.format(ts=datetime.datetime.now(), i=i))
                # Reshape for mir_eval
                voice_est = np.expand_dims(voice_est, 1).T
                voice_patch = voice[i, :, :].T
                mixed_patch = mixed_audio[i, :, :].T
                # Calculate audio quality statistics
                #print('{ts}:\tcalculating metrics {i}.'.format(ts=datetime.datetime.now(), i=i))
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(voice_patch, voice_est, compute_permutation=False)
                sdr_mr, _, _, _ = mir_eval.separation.bss_eval_sources(voice_patch, mixed_patch, compute_permutation=False)
                nsdr = sdr[0] - sdr_mr[0]
                #print('{ts}:\tmetrics calculated {i}.'.format(ts=datetime.datetime.now(), i=i))

                sdrs.append(sdr[0])
                sirs.append(sir[0])
                sars.append(sar[0])
                nsdrs.append(nsdr)
            print('{ts}:\t{f} processed.'.format(ts=datetime.datetime.now(), f=file))

        #  Record mean results for each metric across all batches in the test
        mean_cost = sum(test_costs) / len(test_costs)
        mean_sdr = sum(sdrs) / len(sdrs)
        mean_sir = sum(sirs) / len(sirs)
        mean_sar = sum(sars) / len(sars)
        mean_nsdr = sum(nsdrs) / len(nsdrs)
        metrics.append({'test': test, 'mean_cost': mean_cost, 'mean_sdr': mean_sdr, 'mean_sir': mean_sir,
                        'mean_sar': mean_sar, 'mean_nsdr': mean_nsdr})


    #  Write the results from the experiment to a CSV file, one row per test

    if not os.path.isdir('test_metrics'):
        os.mkdir('test_metrics')
    file_name = 'test_metrics/' + experiment_id + '.csv'
    with open(file_name, 'w') as csvfile:
        fieldnames = ['test', 'mean_cost', 'mean_sdr', 'mean_sir', 'mean_sar', 'mean_nsdr']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for test in metrics:
            writer.writerow(test)

    return metrics


exp_id = str(sys.argv[1])
test_metrics = get_test_metrics(exp_id)
print('{ts}:\nProcessing complete\n{t}'.format(ts=datetime.datetime.now(), t=test_metrics))

