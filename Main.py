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
                    'SAMPLE_RATE': 44100,  # Desired sample rate of audio. Inout will be resampled to this
                    'N_FFT': 1024,  # Number of samples in each fourier transform
                    'FFT_HOP': 256,  # Number of samples between the start of each fourier transform
                    'N_CHANNELS' : 1,  # May be removed - all data is single channel
                    'N_PARALLEL_READERS': 4,
                    'PATCH_WINDOW': 256,
                    'PATCH_HOP': 128,
                    'BATCH_SIZE': 8,
                    'N_SHUFFLE': 20,
                    'EPOCHS': 1
                    }

    experiment_id = np.random.randint(0,1000000)


@ex.capture
def test(sess, data, model, model_config):
    #_, cost = sess.run([model.train_op, model.cost], {model.is_training: False})
    pass




@ex.capture
def train(sess, data, model_config):

    # Build model


    # Summaries
    # Start session
    # Load checkpoint model if required


    # Training finished - save model and close session

    pass


@ex.capture
def optimise():
    pass


@ex.automain
def do_experiment(model_config):

    #tf.reset_default_graph()
    # Prepare data
    print('Preparing dataset')
    train_data, val_data, test_data = Dataset.prepare_datasets(model_config)

    #  Start session
    sess = tf.Session()

    #  Start training
    #model = train(sess, train_data, model_config)

    #  Create iterators
    iterator = train_data.make_initializable_iterator()
    mixed, voice = iterator.get_next()

    training_init_op = iterator.make_initializer(train_data)
    validation_init_op = iterator.make_initializer(val_data)
    testing_init_op = iterator.make_initializer(test_data)

    #  Create variable placeholders
    is_training = tf.placeholder(shape=(), dtype=bool)
    mixed_mag = mixed[0][:, :, 1:, :2]  # Yet more hacking to get around this tuple problem
    mixed_phase = mixed[0][:, :, 1:, 2:]
    voice_mag = voice[0][:, :, 1:, :2]

    #  Build U-Net model
    print('Creating model')
    model = UNet.UNetModel(
        mixed_mag,
        voice_mag,
        mixed_phase,
        is_training
    )
    sess.run(tf.global_variables_initializer())

    #  Begin training loop
    print('Starting training')
    epoch = 1
    iteration = 1
    sess.run(training_init_op)
    while epoch < model_config['EPOCHS'] + 1:

        try:
            _, cost = sess.run([model.train_op, model.cost], {model.is_training: True})
            if iteration % 10 == 0:
                print("            , {0}, {1}".format(iteration, cost))
            iteration += 1
        except tf.errors.OutOfRangeError:
            print('Epoch {e} finished.'.format(e=epoch))
            epoch += 1
            sess.run(training_init_op)
    print('Finished requested number of epochs. Training complete.')

    #  Test trained model
    # Testing - move to test func later
    # Calculate L1 loss
    print('Starting testing')
    sess.run(testing_init_op)
    iteration = 1
    test_costs = list()
    while True:
        try:
            cost = sess.run(model.cost, {model.is_training: False})
            if iteration % 10 == 0:
                print("            , {0}, {1}".format(iteration, cost))
            test_costs.append(cost)
            iteration += 1
        except tf.errors.OutOfRangeError:
            mean_cost = sum(test_costs)/len(test_costs)
            print('Testing complete. Mean cost over test set: {c}'.format(c=mean_cost))
            break
    # Calculate audio loss metrics



    pass
