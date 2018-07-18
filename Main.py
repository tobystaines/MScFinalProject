import numpy as np
import tensorflow as tf
from sacred import Experiment
import pickle
import os

import Audio_functions as af
import Model_functions as mf
import UNet
import Dataset
import Utils

ex = Experiment('UNet_Speech_Separation')

@ex.config
def cfg():
    model_config = {"model_base_dir": "checkpoints",  # Base folder for model checkpoints
                    "log_dir": "logs",  # Base folder for log files
                    "data_root": 'C:/Users/Toby/Jupyter Notebooks/My Work/MSc Project/Test Audio/GANdatasetsMini/',  # Base folder of CHiME 3 dataset
                    'SAMPLE_RATE': 44100,  # Desired sample rate of audio. Input will be resampled to this
                    'N_FFT': 1024,  # Number of samples in each fourier transform
                    'FFT_HOP': 256,  # Number of samples between the start of each fourier transform
                    'N_CHANNELS' : 1,  # May be removed - all data is single channel
                    'N_PARALLEL_READERS': 4,
                    'PATCH_WINDOW': 256,
                    'PATCH_HOP': 128,
                    'BATCH_SIZE': 8,
                    'N_SHUFFLE': 20,
                    'EPOCHS': 1,  # Number of full passes through the dataset to train for
                    'EARLY_STOPPING': True,  # Should validation data checks be used for early stopping?
                    'VAL_ITERS': 15,  # Number of training iterations between validation checks,
                    'NUM_WORSE_VAL_CHECKS': 2  # Number of successively worse validation checks before early stopping
                    }

    experiment_id = np.random.randint(0,1000000)


@ex.capture
def train(sess, model, model_config, handle, training_iterator, training_handle, validation_iterator, validation_handle, writer):

    #  Begin training loop
    print('Starting training')
    epoch = 1
    iteration = 1
    min_val_cost = 1
    worse_val_checks = 0
    sess.run(training_iterator.initializer)
    # Train for the specified number of epochs, unless early stopping is triggered
    while epoch < model_config['EPOCHS'] + 1 and worse_val_checks < model_config['NUM_WORSE_VAL_CHECKS']:
        try:
            _, cost = sess.run([model.train_op, model.cost], {model.is_training: True, handle: training_handle})
            if iteration % 10 == 0:
                print("       Training iteration: {i}, Loss: {l}".format(i=iteration, l=cost))
            writer.add_summary(cost, iteration)
            # If using early stopping, enter validation loop
            if model_config['EARLY_STOPPING'] and iteration % model_config['VAL_ITERS'] == 0:
                print('Validating')
                sess.run(validation_iterator.initializer)
                val_costs = list()
                while True:
                    try:
                        val_cost = sess.run(model.cost, {model.is_training: False, handle: validation_handle})
                        val_costs.append(val_cost)
                    except tf.errors.OutOfRangeError:
                        val_check_mean_cost = sum(val_costs) / len(val_costs)
                        print('Validation check mean loss: {l}'.format(l=val_check_mean_cost))
                        if val_check_mean_cost > min_val_cost:  # If validation loss has worsened, take note
                            worse_val_checks += 1
                            print('Validation loss has worsened. worse_val_checks = {w}'.format(w=worse_val_checks))
                        else:  # If validation cost has improved, reset counter
                            min_val_cost = val_check_mean_cost
                            worse_val_checks = 0
                            print('Validation loss has improved!')
                        break
            iteration += 1
        # When the dataset is exhausted, note the end of the epoch
        except tf.errors.OutOfRangeError:
            print('Epoch {e} finished.'.format(e=epoch))
            epoch += 1
            sess.run(training_iterator.initializer)
    print('Finished requested number of epochs. Training complete.')

    return model


@ex.capture
def test(sess, model, handle, testing_iterator, testing_handle):
    # Testing - move to test func later
    # Calculate L1 loss
    print('Starting testing')
    sess.run(testing_iterator.initializer)
    iteration = 1
    test_costs = list()
    while True:
        try:
            cost = sess.run(model.cost, {model.is_training: False, handle: testing_handle})
            if iteration % 10 == 0:
                print("       Testing iteration: {i}, Loss: {l}".format(i=iteration, l=cost))
            test_costs.append(cost)
            iteration += 1
        except tf.errors.OutOfRangeError:
            mean_cost = sum(test_costs) / len(test_costs)
            print('Testing complete. Mean loss over test set: {l}'.format(l=mean_cost))
            break
            # Calculate audio loss metrics
    return mean_cost


@ex.capture
def optimise():
    pass


@ex.automain
def do_experiment(model_config, experiment_id):

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
    mixed_mag = mixed[0][:, :, 1:, :2]  # Yet more hacking to get around this tuple problem
    mixed_phase = mixed[0][:, :, 1:, 2:]
    voice_mag = voice[0][:, :, 1:, :2]

    # Build U-Net model
    print('Creating model')
    model = UNet.UNetModel(mixed_mag, voice_mag, mixed_phase, is_training)

    # Summaries
    model_folder = str(experiment_id)
    tf.summary.scalar('L1_loss', model.cost)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(model_config["log_dir"], model_folder), graph=sess.graph)

    # Train the model
    sess.run(tf.global_variables_initializer())
    model = train(sess, model, model_config, handle, training_iterator, training_handle, validation_iterator, validation_handle, writer)

    # Test trained model
    mean_test_loss = test(sess, model, handle, testing_iterator, testing_handle)
    print(mean_test_loss, '\nAll done!')



