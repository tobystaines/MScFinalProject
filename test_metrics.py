



def get_test_metrics(experiment_id):

    dump_folder = 'dumps/' + experiment_id


    """
    test_costs = list()
    sdrs = list()
    sirs = list()
    sars = list()
    nsdrs = list()
    """
    """

    for i in range(voice_est_mag.shape[0]):
        # Transform output back to audio
        print('{ts}:\tConverting spectrogram to audio'.format(ts=datetime.datetime.now()))
        voice_est = af.spectrogramToAudioFile(np.squeeze(voice_est_mag[i, :, :, :]).T, model_config['N_FFT'],
                                              model_config['FFT_HOP'], phase=np.squeeze(mixed_phase[i, :, :, :]).T)
        # Reshape for mir_eval
        voice_est = np.expand_dims(voice_est, 1).T
        voice_patch = voice[i, :, :].T
        mixed_patch = mixed_audio[i, :, :].T
        # Calculate audio quality statistics
        print('{ts}:\tCalculating audio quality metrics'.format(ts=datetime.datetime.now()))
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(voice_patch, voice_est, compute_permutation=False)
        sdr_mr, _, _, _ = mir_eval.separation.bss_eval_sources(voice_patch, mixed_patch, compute_permutation=False)
        nsdr = sdr[0] - sdr_mr[0]
        sdrs.append(sdr[0])
        sirs.append(sir[0])
        sars.append(sar[0])
        nsdrs.append(nsdr)

    """
    """
                mean_cost = sum(test_costs) / len(test_costs)
                mean_sdr = sum(sdrs) / len(sdrs)
                mean_sir = sum(sirs) / len(sirs)
                mean_sar = sum(sars) / len(sars)
                mean_nsdr = sum(nsdrs) / len(nsdrs)
                for var in [(mean_cost, 'mean_cost'), (mean_sdr, 'mean_sdr'), (mean_sir, 'mean_sir'),
                            (mean_sar, 'mean_sar'), (mean_nsdr, 'mean_nsdr')]:
                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag='Testing_{v}'.format(v=var[1]), simple_value=var[0])])
                    writer.add_summary(summary, test_count)
                print('Testing complete. Mean results over test set:\n'
                      'Loss: {mc}\n'
                      'SDR:  {sdr}\n'
                      'SIR:  {sir}\n'
                      'SAR:  {sar}\n'
                      'NSDR: {nsdr}'.format(mc=mean_cost, sdr=mean_sdr, sir=mean_sir, sar=mean_sar, nsdr=mean_nsdr))
                """