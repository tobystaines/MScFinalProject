import os
import tensorflow as tf
import Audio_functions as af
import UNet
import Dataset

#  Locate data to use
path = 'C:/Users/Toby/MSc_Project/Test_Audio/GANdatasetsMini/train_sup/Mixed/F01_22GC010A_BUS.CH1.wav'

#  Locate model checkpoint to load
local = True
checkpoint = '8/8-13'

if local:
    model_base_dir = 'C:/Users/Toby/MSc_Project/MScFinalProjectCheckpoints'
    data_folder = 'C:/Users/Toby/MSc_Project/Test_Audio/GANdatasetsMini/train_sup/Mixed'
else:
    model_base_dir = '/home/enterprise.internal.city.ac.uk/acvn728/checkpoints'
    data_folder = None

# Prepare data pipeline
data = Dataset.get_dataset(data_folder=data_folder,
                           sample_rate=16000,
                           n_fft=1024,
                           fft_hop=256,
                           n_channels=1,
                           patch_window=256,
                           patch_hop=128,
                           n_parallel_readers=4,
                           normalise=False)

data = Dataset.zip_datasets(data, data, 1)

#  Build model structure and load weights
tf.reset_default_graph()
sess = tf.Session()

mixed, voice = data.make_one_shot_iterator().get_next()

# Create variable placeholders
is_training = tf.placeholder(shape=(), dtype=bool)
mixed_mag = tf.expand_dims(mixed[0][:, :, 1:, 0], 3)  # Yet more hacking to get around this tuple problem
mixed_phase = tf.expand_dims(mixed[0][:, :, 1:, 1], 3)
voice_mag = tf.expand_dims(voice[0][:, :, 1:, 0], 3)

# Build U-Net model
print('Creating model')
model = UNet.UNetModel(mixed_mag, voice_mag, mixed_phase, 'unet', is_training, name='U_Net_Model')

print('Loading checkpoint')
checkpoint = os.path.join(model_base_dir, model_config['checkpoint_to_load'])
restorer = tf.train.Saver()
restorer.restore(sess, checkpoint)