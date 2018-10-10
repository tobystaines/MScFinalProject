import tensorflow as tf
import model_functions as mf
from SegCaps import capsule_layers
from keras import layers


class MagnitudeModel(object):
    """
    Top level U-Net object.
    Attributes:
        mixed_mag: Input placeholder for magnitude spectrogram of mixed signals (voice plus background noise) - X
        voice_mag: Input placeholder for magnitude spectrogram of isolated voice signal - Y
        mixed_phase: Input placeholder for phase spectrogram of mixed signals (voice plus background noise)
        mixed_audio: Input placeholder for waveform audio of mixed signals (voice plus background noise)
        voice_audio: Input placeholder for waveform audio of isolated voice signal
        variant: The type of U-Net model (Normal convolutional or capsule based)
        is_training: Boolean - should the model be trained on the current input or not
        name: Model instance name
    """
    def __init__(self, mixed_mag, voice_mag, mixed_phase, mixed_audio, voice_audio, variant, is_training, learning_rate,
                 name):
        with tf.variable_scope(name):
            self.mixed_mag = mixed_mag
            self.voice_mag = voice_mag
            self.mixed_phase = mixed_phase
            self.mixed_audio = mixed_audio
            self.voice_audio = voice_audio
            self.variant = variant
            self.is_training = is_training

            if self.variant in ['unet', 'capsunet']:
                self.voice_mask_network = UNet(mixed_mag, variant, is_training=is_training, reuse=False, name='voice-mask-unet')
            elif self.variant == 'basic_capsnet':
                self.voice_mask_network = BasicCapsnet(mixed_mag, name='SegCaps_CapsNetBasic')

            self.voice_mask = self.voice_mask_network.output

            self.gen_voice = self.voice_mask * mixed_mag

            self.cost = mf.l1_loss(self.gen_voice, voice_mag)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.5,
            )
            self.train_op = self.optimizer.minimize(self.cost)
            self.check_op = tf.add_check_numerics_ops()


class UNet(object):

    def __init__(self, input_tensor, variant, is_training, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            self.variant = variant

            if self.variant == 'unet':
                self.encoder = UNetEncoder(input_tensor, is_training, reuse)
                self.decoder = UNetDecoder(self.encoder.output, self.encoder, is_training, reuse)
            elif self.variant == 'capsunet':
                self.encoder = CapsUNetEncoder(input_tensor, is_training, reuse)
                self.decoder = CapsUNetDecoder(self.encoder.output, self.encoder, is_training, reuse)

            self.output = mf.tanh(self.decoder.output) / 2 + .5


class UNetEncoder(object):
    """
    The down-convolution side of a convoltional U-Net model.
    """

    def __init__(self, input_tensor, is_training, reuse):
        net = input_tensor
        with tf.variable_scope('encoder'):
            with tf.variable_scope('layer-1'):
                net = mf.conv(net, filters=16, kernel_size=5, stride=(2, 2))
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=32, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=64, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l3 = net

            with tf.variable_scope('layer-4'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=128, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l4 = net

            with tf.variable_scope('layer-5'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=256, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.l5 = net

            with tf.variable_scope('layer-6'):
                net = mf.lrelu(net)
                net = mf.conv(net, filters=512, kernel_size=5, stride=(2, 2))

            self.output = net


class UNetDecoder(object):
    """
    The up-convolution side of a convolutional U-Net model
    """
    def __init__(self, input_tensor, encoder, is_training, reuse):
        net = input_tensor

        with tf.variable_scope('decoder'):
            with tf.variable_scope('layer-1'):
                net = mf.relu(net)
                net = mf.deconv(net, filters=256, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                net = mf.dropout(net, .5)

            with tf.variable_scope('layer-2'):
                net = mf.relu(mf.concat(net, encoder.l5))
                net = mf.deconv(net, filters=128, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                net = mf.dropout(net, .5)

            with tf.variable_scope('layer-3'):
                net = mf.relu(mf.concat(net, encoder.l4))
                net = mf.deconv(net, filters=64, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                net = mf.dropout(net, .5)

            with tf.variable_scope('layer-4'):
                net = mf.relu(mf.concat(net, encoder.l3))
                net = mf.deconv(net, filters=32, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)

            with tf.variable_scope('layer-5'):
                net = mf.relu(mf.concat(net, encoder.l2))
                net = mf.deconv(net, filters=16, kernel_size=5, stride=(2, 2))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)

            with tf.variable_scope('layer-6'):
                net = mf.relu(mf.concat(net, encoder.l1))
                net = mf.deconv(net, filters=1, kernel_size=5, stride=(2, 2))

            self.output = net


class CapsUNetEncoder(object):
    """
    The down-convolutional side of a capsule based U-Net model.
    """
    def __init__(self, input_tensor, is_training, reuse):
        net = input_tensor

        self.output = net


class CapsUNetDecoder(object):
    """
    The up-convolutional side of a capsule based U-Net model.
    """
    def __init__(self, input_tensor, encoder, is_training, reuse):
        net = input_tensor

        self.output = net


class BasicCapsnet(object):

    def __init__(self, mixed_mag, name):
        """
        input_tensor: Tensor with shape [batch_size, height, width, channels]
        is_training:  Boolean - should the model be trained on the current input or not
        name:         Model instance name
        """
        with tf.variable_scope(name):
            self.mixed_mag = mixed_mag

            with tf.variable_scope('Convolution'):
                net = mf.conv(mixed_mag, filters=128, kernel_size=5, stride=(1, 1))

                # Reshape layer to be 1 capsule x [filters] atoms
                _, H, W, C = net.get_shape()
                net = layers.Reshape((H.value, W.value, 1, C.value))(net)
                self.conv1 = net

            with tf.variable_scope('Primary_Caps'):
                net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=8, strides=1,
                                                      padding='same',
                                                      routings=1, name='primarycaps')(net)
                self.primary_caps = net

            with tf.variable_scope('Seg_Caps'):
                net = capsule_layers.ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=8, strides=1,
                                                      padding='same',
                                                      routings=3, name='seg_caps')(net)
                self.seg_caps = net

            with tf.variable_scope('Reconstruction'):
                net = capsule_layers.ConvCapsuleLayer(kernel_size=1, num_capsule=1, num_atoms=1, strides=1,
                                                      padding='same',
                                                      routings=3, name='reconstruction')(net)
                net = tf.squeeze(net, -1)

            self.output = net


class ComplexNumberModel(object):
    """
    Top level U-Net object.
    Attributes:
        mixed_mag: Input placeholder for magnitude spectrogram of mixed signals (voice plus background noise) - X
        voice_mag: Input placeholder for magnitude spectrogram of isolated voice signal - Y
        mixed_phase: Input placeholder for phase spectrogram of mixed signals (voice plus background noise)
        mixed_audio: Input placeholder for waveform audio of mixed signals (voice plus background noise)
        voice_audio: Input placeholder for waveform audio of isolated voice signal
        variant: The type of U-Net model (Normal convolutional or capsule based)
        is_training: Boolean - should the model be trained on the current input or not
        name: Model instance name
    """

    def __init__(self, mixed_spec, voice_spec, mixed_audio, voice_audio, variant, is_training, learning_rate,
                 name='complex_unet_model'):
        with tf.variable_scope(name):
            self.mixed_spec = mixed_spec
            self.voice_spec = voice_spec

            self.input_shape = mixed_spec.get_shape().as_list()
            self.mixed_audio = mixed_audio
            self.voice_audio = voice_audio
            self.variant = variant
            self.is_training = is_training

            self.voice_mask_unet = ComplexUNet(mixed_spec, variant, is_training=is_training, reuse=False,
                                               name='voice-mask-unet')

            self.voice_mask = self.voice_mask_unet.output

            self.gen_voice = self.voice_mask * mixed_spec

            self.cost = mf.l1_loss(self.gen_voice, voice_spec)

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.5,
            )
            self.train_op = self.optimizer.minimize(self.cost)


class ComplexUNet(object):

    def __init__(self, input_tensor, variant, is_training, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            self.variant = variant

            if self.variant == 'unet':
                self.encoder = UNetEncoder(input_tensor, is_training, reuse)
                self.decoder = UNetDecoder(self.encoder.output, self.encoder, is_training, reuse)
            elif self.variant == 'capsunet':
                self.encoder = ComplexEncoder(input_tensor)
                self.decoder = ComplexDecoder(self.encoder.output, self.encoder)

            self.output = mf.tanh(self.decoder.output) / 2 + .5


class ComplexEncoder(object):
    def __init__(self, input_tensor):
        net = input_tensor
        with tf.variable_scope('encoder'):
            with tf.variable_scope('layer-1'):
                # Reshape layer to be 1 capsule x [filters] atoms
                _, self.H, self.W, self.C = net.get_shape()
                net = layers.Reshape((self.H.value, self.W.value, 1, self.C.value))(net)
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=2, strides=2,
                                                      padding='same',
                                                      routings=1)(net)
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=32, num_atoms=2, strides=2,
                                                      padding='same',
                                                      routings=1)(net)

        self.output = net


class ComplexDecoder(object):
    def __init__(self, input_tensor, encoder):
        net = input_tensor
        with tf.variable_scope('decoder'):
            with tf.variable_scope('layer-1'):
                net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=2, upsamp_type='deconv',
                                                        scaling=2, padding='same', routings=3)(net)
                net = layers.Concatenate(axis=-2)([net, encoder.l2])
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=2, upsamp_type='deconv',
                                                        scaling=2, padding='same', routings=3)(net)
                net = layers.Concatenate(axis=-2)([net, encoder.l1])
                net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=2, upsamp_type='deconv',
                                                        scaling=1, padding='same', routings=3)(net)
                net = layers.Reshape((encoder.H.value, encoder.W.value, encoder.C.value))(net)
            self.output = net
