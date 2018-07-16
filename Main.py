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
                    'SAMPLE_RATE': 44100,  # Desired sample rate of audio. Inout will be resampled to this
                    'N_FFT': 1024,  # Number of samples in each fourier transform
                    'FFT_HOP': 256,  # Number of samples between the start of each fourier transform
                    'N_CHANNELS' : 1,  # May be removed - all data is single channel
                    'N_PARALLEL_READERS': 4,
                    'PATCH_WINDOW': 256,
                    'PATCH_HOP': 128,
                    'BATCH_SIZE': 8,
                    'N_SHUFFLE': 20
                    }

    experiment_id = np.random.randint(0,1000000)


@ex.capture
def test():
    pass


@ex.capture
def train():
    pass


@ex.capture
def optimise():
    pass


@ex.automain
def do_experiment():

    # Prepare data for input
    if os.path.exists('dataset.pkl'):  # Load existing dataset
        with open('dataset.pkl', 'r') as file:
            dataset = pickle.load(file)
        print("Loaded dataset from pickle!")
    else:  # Create new dataset
        train, _ = Dataset.getCHiME3('train')
        validation = list(Dataset.getCHiME3('validation'))
        test = list(Dataset.getCHiME3('test'))

        data = dict()
        data["train"] = train
        data["valid"] = validation
        data["test"] = test


        # Zip up all paired dataset partitions so we have (mixture, voice) tuples
        data["train"] = zip(data["train"][0], data["train"][1])
        data["valid"] = zip(data["valid"][0], data["valid"][1])
        data["test"] = zip(data["test"][0], data["test"][1])

        with open('dataset.pkl', 'wb') as file:
            pickle.dump(data, file)
        print("Created dataset structure")

    # Train the network

    pass