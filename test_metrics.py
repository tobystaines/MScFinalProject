import sys
import csv
import pickle
import datetime
from glob import glob
import numpy as np
import mir_eval
import Audio_functions as af


def get_test_metrics(experiment_id):

    # Calculate number of test runs in experiment
    dump_folder = 'dumps/' + experiment_id
    file_list = glob(dump_folder + '/*')
    test_num = max([int(file.split('_')[2]) for file in file_list]) + 1
    metrics = list()
    for test in range(test_num):
        print('{ts}:\tProcessing test {t}'.format(ts=datetime.datetime.now(), t=test))
        test_files = [file for file in file_list if file.split('_')[2] == str(test)]
        test_costs = list()
        sdrs = list()
        sirs = list()
        sars = list()
        nsdrs = list()

        for file in test_files:
            cost, voice_est_mag, voice, mixed_audio, mixed_phase, model_config = pickle.load(open(file, 'rb'))
            test_costs.append(cost)
            for i in range(voice_est_mag.shape[0]):
                # Transform output back to audio
                voice_est = af.spectrogramToAudioFile(np.squeeze(voice_est_mag[i, :, :, :]).T, model_config['N_FFT'],
                                                      model_config['FFT_HOP'], phase=np.squeeze(mixed_phase[i, :, :, :]).T)
                # Reshape for mir_eval
                voice_est = np.expand_dims(voice_est, 1).T
                voice_patch = voice[i, :, :].T
                mixed_patch = mixed_audio[i, :, :].T
                # Calculate audio quality statistics
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(voice_patch, voice_est, compute_permutation=False)
                sdr_mr, _, _, _ = mir_eval.separation.bss_eval_sources(voice_patch, mixed_patch, compute_permutation=False)
                nsdr = sdr[0] - sdr_mr[0]

                sdrs.append(sdr[0])
                sirs.append(sir[0])
                sars.append(sar[0])
                nsdrs.append(nsdr)

        mean_cost = sum(test_costs) / len(test_costs)
        mean_sdr = sum(sdrs) / len(sdrs)
        mean_sir = sum(sirs) / len(sirs)
        mean_sar = sum(sars) / len(sars)
        mean_nsdr = sum(nsdrs) / len(nsdrs)
        metrics.append({'test': test, 'mean_cost': mean_cost, 'mean_sdr': mean_sdr, 'mean_sir': mean_sir,
                        'mean_sar': mean_sar, 'mean_nsdr': mean_nsdr})

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
print(test_metrics)

