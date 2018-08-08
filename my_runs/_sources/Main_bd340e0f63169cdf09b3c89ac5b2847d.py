import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
import mir_eval

import sys
import os
import errno

import Audio_functions as af
import UNet
import Dataset

assert sys.version_info >= (3, 5)
sys.path.append('/home/enterprise.internal.city.ac.uk/acvn728/.local/lib/python3.5/site')
ex = Experiment('UNet_Speech_Separation', interactive=True)
ex.observers.append(FileStorageObserver.create('my_runs'))


@ex.config
def cfg():
    model_config = {"saving": True,  # Whether to take checkpoints
                    "loading": False,  # Whether to load an existing checkpoint
                    "local_run": True,  # Whether experiment is running on laptop or server
                    "checkpoint_to_load": "196363/196363-1801",
                    'SAMPLE_RATE': 16000,  # Desired sample rate of audio. Input will be resampled to this
                    'N_FFT': 1024,  # Number of samples in each fourier transform
                    'FFT_HOP': 256,  # Number of samples between the start of each fourier transform
                    'N_CHANNELS' : 1,  # May be removed - all data is single channel
                    'N_PARALLEL_READERS': 4,
                    'PATCH_WINDOW': 256,
                    'PATCH_HOP': 128,
                    'BATCH_SIZE': 5,
                    'N_SHUFFLE': 5,
                    'EPOCHS': 3,  # Number of full passes through the dataset to train for
                    'EARLY_STOPPING': False,  # Should validation data checks be used for early stopping?
                    'VAL_ITERS': 200,  # Number of training iterations between validation checks,
                    'NUM_WORSE_VAL_CHECKS': 3,  # Number of successively worse validation checks before early stopping,
                    'NORMALISE_MAG': False
                    }

    if model_config['local_run']:  # Data and Checkpoint directories on my laptop
        model_config['data_root'] = 'C:/Users/Toby/MSc_Project/Test_Audio/GANdatasetsMini/'
        model_config['model_base_dir'] = 'C:/Users/Toby/MSc_Project/MScFinalProjectCheckpoints'
        model_config['log_dir'] = 'logs/local'

    else:  # Data and Checkpoint directories on the uni server
        model_config['data_root'] = '/data/CHiME3/data/audio/16kHz/isolated/'
        model_config['model_base_dir'] = '/home/enterprise.internal.city.ac.uk/acvn728/checkpoints'
        model_config['log_dir'] = 'logs/ssh'


@ex.capture
def train(sess, model, model_config, model_folder, handle, training_iterator, training_handle, validation_iterator,
          validation_handle, writer):

    def validation(min_val_cost, worse_val_checks, best_model, iteration):
        print('Validating')
        sess.run(validation_iterator.initializer)
        val_costs = list()
        val_iteration = 1
        while True:
            try:
                val_cost = sess.run(model.cost, {model.is_training: False, handle: validation_handle})
                val_costs.append(val_cost)
                if val_iteration % 50 == 0:
                    print("       Validation iteration: {i}, Loss: {vc}".format(i=val_iteration, vc=val_cost))
                val_iteration += 1
            except tf.errors.OutOfRangeError:
                # Calculate and record mean loss over validation dataset
                val_check_mean_cost = sum(val_costs) / len(val_costs)
                print('Validation check mean loss: {l}'.format(l=val_check_mean_cost))
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='Validation_mean_loss', simple_value=val_check_mean_cost)])
                writer.add_summary(summary, val_check)
                # If validation loss has worsened increment the counter, else, reset the counter
                if val_check_mean_cost > min_val_cost:
                    worse_val_checks += 1
                    print('Validation loss has worsened. worse_val_checks = {w}'.format(w=worse_val_checks))
                else:
                    min_val_cost = val_check_mean_cost
                    best_model = model
                    worse_val_checks = 0
                    print('Validation loss has improved!')

                break
        return min_val_cost, worse_val_checks, best_model

    print('Starting training')
    # Initialise variables and define summary
    epoch = 1
    iteration = 1
    min_val_cost = 1
    val_check = 1
    worse_val_checks = 0
    best_model = model
    training_summary = tf.summary.scalar('Training_loss', model.cost)
    mix_summary = tf.summary.image('Mixture', model.mixed)
    voice_summary = tf.summary.image('Voice', model.voice)
    mask_summary = tf.summary.image('Voice_Mask', model.voice_mask)
    gen_voice_summary = tf.summary.image('Generated_Voice', model.gen_voice)
    saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2)
    sess.run(training_iterator.initializer)

    # Begin training loop
    # Train for the specified number of epochs, unless early stopping is triggered
    while epoch < model_config['EPOCHS'] + 1 and worse_val_checks < model_config['NUM_WORSE_VAL_CHECKS']:
        try:
            _, cost, summary, mix, voice, mask, gen_voice = sess.run([model.train_op, model.cost, training_summary,
                                                                     mix_summary, voice_summary, mask_summary,
                                                                     gen_voice_summary], {model.is_training: True,
                                                                                         handle: training_handle})
            if iteration % 20 == 0:
                print("       Training iteration: {i}, Loss: {c}".format(i=iteration, c=cost))
                writer.add_summary(summary, iteration)  # Record the loss at each iteration
                writer.add_summary(mix, iteration)
                writer.add_summary(voice, iteration)
                writer.add_summary(mask, iteration)
                writer.add_summary(gen_voice, iteration)
            # If using early stopping, enter validation loop
            if model_config['EARLY_STOPPING'] and iteration % model_config['VAL_ITERS'] == 0:
                min_val_cost, worse_val_checks, best_model = validation(min_val_cost, worse_val_checks, best_model, iteration)
                val_check += 1

            iteration += 1

        # When the dataset is exhausted, note the end of the epoch
        except tf.errors.OutOfRangeError:
            print('Epoch {e} finished.'.format(e=epoch))
            epoch += 1
            sess.run(training_iterator.initializer)

    if model_config['EARLY_STOPPING'] and worse_val_checks >= model_config['NUM_WORSE_VAL_CHECKS']:
        print('Stopped early due to validation criteria.')
    else:
        # Final validation check
        if iteration % model_config['VAL_ITERS'] != 1 or not model_config['EARLY_STOPPING']:
            min_val_cost, _, best_model = validation(min_val_cost, worse_val_checks, best_model, iteration)
        print('Finished requested number of epochs. Training complete.')
    print('Best validation loss: {mvc}'.format(mvc=min_val_cost))
    model = best_model

    if model_config['saving']:
        # Make sure there is a folder to save the checkpoint in
        checkpoint_path = os.path.join(model_config["model_base_dir"], model_folder)
        try:
            os.makedirs(checkpoint_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        # Save the final best model
        print('Checkpoint')
        saver.save(sess, os.path.join(checkpoint_path, model_folder), global_step=int(iteration))

    return model


@ex.capture
def test(sess, model, model_config, handle, testing_iterator, testing_handle, writer, test_count):

    # Calculate L1 loss
    print('Starting testing')
    sess.run(testing_iterator.initializer)
    iteration = 1
    test_costs = list()
    sdrs = list()
    sirs = list()
    sars = list()
    while True:
        try:
            cost, voice_est_mag, voice_mag, mixed_phase = sess.run([model.cost, model.gen_voice, model.voice,
                                                                    model.mixed_phase], {model.is_training: False,
                                                                                         handle: testing_handle})
            if iteration % 50 == 0:
                print("       Testing iteration: {i}, Loss: {c}".format(i=iteration, c=cost))
            test_costs.append(cost)
            # Transform output back to audio
            for i in range(voice_mag.shape[0]):
                voice_est = af.spectrogramToAudioFile(np.squeeze(voice_est_mag[i, :, :, :]).T, model_config['N_FFT'],
                                                      model_config['FFT_HOP'], phase=np.squeeze(mixed_phase[i, :, :, :]).T)
                # Should we use voice or the original audio? (Might be hard to split into matching patches)
                voice = af.spectrogramToAudioFile(np.squeeze(voice_mag[i, :, :, :]).T, model_config['N_FFT'],
                                                  model_config['FFT_HOP'], phase=np.squeeze(mixed_phase[i, :, :, :]).T)
                # TODO Pad to ensure equal length?
                # Reshape for mir_eval
                voice_est = np.expand_dims(voice_est, 1).T
                voice = np.expand_dims(voice, 1).T
                # Calculate audio quality statistics
                sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(voice, voice_est)
                sdrs.append(sdr[0])
                sirs.append(sir[0])
                sars.append(sar[0])
            iteration += 1
        except tf.errors.OutOfRangeError:
            # At the end of the dataset, calculate, record and print mean results
            mean_cost = sum(test_costs) / len(test_costs)
            mean_sdr = sum(sdrs) / len(sdrs)
            mean_sir = sum(sirs) / len(sirs)
            mean_sar = sum(sars) / len(sars)
            for var in [(mean_cost, 'mean_cost'), (mean_sdr, 'mean_sdr'), (mean_sir, 'mean_sir'), (mean_sar, 'mean_sar')]:
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='Testing_{v}'.format(v=var[1]), simple_value=var[0])])
                writer.add_summary(summary, test_count)
            print('Testing complete. Mean results over test set:\n'
                  'Loss: {mc}\n'
                  'SDR:  {sdr}\n'
                  'SIR:  {sir}\n'
                  'SAR:  {sar}'.format(mc=mean_cost, sdr=mean_sdr, sir=mean_sir, sar=mean_sar))
            break

    return mean_cost


@ex.automain
def do_experiment(model_config):

    tf.reset_default_graph()
    experiment_id = ex.current_run._id
    print('Experiment ID: {eid}'.format(eid=experiment_id))

    # Prepare data
    print('Preparing dataset')
    train_data, val_data, test_data = Dataset.prepare_datasets(model_config)

    # Start session
    sess = tf.Session()

    # Create iterators
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, train_data.output_types, train_data.output_shapes)
    mixed, voice = iterator.get_next()

    training_iterator = train_data.make_initializable_iterator()
    validation_iterator = val_data.make_initializable_iterator()
    testing_iterator = test_data.make_initializable_iterator()

    training_handle = sess.run(training_iterator.string_handle())
    validation_handle = sess.run(validation_iterator.string_handle())
    testing_handle = sess.run(testing_iterator.string_handle())

    # Create variable placeholders
    is_training = tf.placeholder(shape=(), dtype=bool)
    mixed_mag = tf.expand_dims(mixed[0][:, :, 1:, 0], 3)  # Yet more hacking to get around this tuple problem
    mixed_phase = tf.expand_dims(mixed[0][:, :, 1:, 1], 3)
    voice_mag = tf.expand_dims(voice[0][:, :, 1:, 0], 3)

    # Build U-Net model
    print('Creating model')
    model = UNet.UNetModel(mixed_mag, voice_mag, mixed_phase, 'unet', is_training, name='U_Net_Model')

    if model_config['loading']:
        # TODO - Think this works now but needs proper testing
        print('Loading checkpoint')
        checkpoint = os.path.join(model_config['model_base_dir'], model_config['checkpoint_to_load'])
        restorer = tf.train.Saver()
        restorer.restore(sess, checkpoint)

    # Summaries
    model_folder = str(experiment_id)
    writer = tf.summary.FileWriter(os.path.join(model_config["log_dir"], model_folder), graph=sess.graph)

    # Get baseline metrics at initialisation
    sess.run(tf.global_variables_initializer())
    print('Running initialisation test')
    test_count = 1
    initial_test_loss = test(sess, model, model_config, handle, testing_iterator, testing_handle, writer, test_count)
    test_count += 1

    # Train the model
    model = train(sess, model, model_config, model_folder, handle, training_iterator, training_handle,
                  validation_iterator, validation_handle, writer)

    # Test trained model
    mean_test_loss = test(sess, model, model_config, handle, testing_iterator, testing_handle, writer, test_count)
    print('All done!\nInitial test loss: {init}\nFinal test loss: {final}'
          .format(init=initial_test_loss, final=mean_test_loss))



