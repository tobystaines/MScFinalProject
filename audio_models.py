import tensorflow as tf
import model_functions as mf
from SegCaps import capsule_layers
from keras import layers



class MagnitudeModel(object):
    """
    Top level object for models working on magnitude spectrograms.
    Attributes:
        mixed_mag: Input placeholder for magnitude spectrogram of mixed signals (voice plus background noise) - X
        voice_mag: Input placeholder for magnitude spectrogram of isolated voice signal - Y
        mixed_phase: Input placeholder for phase spectrogram of mixed signals (voice plus background noise)
        mixed_audio: Input placeholder for waveform audio of mixed signals (voice plus background noise)
        voice_audio: Input placeholder for waveform audio of isolated voice signal
        variant: The type of U-Net model (Normal convolutional or capsule based)
        is_training: Boolean - should the model be trained on the current input or not
        learning_rate: The learning rate the model should be trained with.
        name: Model instance name
    """
    def __init__(self, mixed_input, voice_input, mixed_phase, mixed_audio, voice_audio, background_audio,
                 variant, is_training, learning_rate, data_type, name):
        with tf.variable_scope(name):
            self.mixed_input = mixed_input
            self.voice_input = voice_input
            self.mixed_phase = mixed_phase
            self.mixed_audio = mixed_audio
            self.voice_audio = voice_audio
            self.background_audio = background_audio
            self.variant = variant
            self.is_training = is_training

            if self.variant in ['unet', 'capsunet']:
                self.voice_mask_network = UNet(mixed_input, variant, data_type, is_training=is_training, reuse=False, name='voice-mask-unet')
            elif self.variant == 'basic_capsnet':
                self.voice_mask_network = BasicCapsnet(mixed_input, name='SegCaps_CapsNetBasic')
            elif self.variant == 'conv_net':
                self.voice_mask_network = conv_net(mixed_input, is_training=is_training, reuse=None, name='basic_cnn')

            self.voice_mask = self.voice_mask_network.output

            if data_type == 'mag':
                self.gen_voice = self.voice_mask * mixed_input
                self.cost = mf.l1_loss(self.gen_voice, voice_input)

            elif data_type in ['mag_phase', 'mag_phase_real_imag']:
                self.gen_voice = self.voice_mask * mixed_input
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 1]) * 0.00001
                self.cost = (self.mag_loss + self.phase_loss)/2

            elif data_type == 'mag_phase_diff':
                self.gen_voice_mag = tf.expand_dims(self.voice_mask[:, :, :, 0] * mixed_input[:, :, :, 0], axis=3)
                self.mag_loss = mf.l1_loss(self.gen_voice_mag[:, :, :, 0], voice_input[:, :, :, 0])
                self.phase_loss = mf.l1_phase_loss(mf.phase_difference(mixed_input[:, :, :, 1], voice_input[:, :, :, 1]),
                                                   self.voice_mask[:, :, :, 1]) * 0.00001
                self.cost = (self.mag_loss + self.phase_loss) / 2
                self.gen_voice_phase = tf.expand_dims(self.voice_mask[:, :, :, 1] + mixed_input[:, :, :, 1], axis=3)
                self.gen_voice = tf.concat((self.gen_voice_mag, self.gen_voice_phase), axis=3)

            elif data_type == 'real_imag':
                self.gen_voice = self.voice_mask * mixed_input
                self.real_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.imag_loss = mf.l1_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 1])
                self.cost = (self.real_loss + self.imag_loss)/2

            elif data_type == 'mag_real_imag':
                self.gen_voice = self.voice_mask * mixed_input
                self.mag_loss = mf.l1_loss(self.gen_voice[:, :, :, 0], voice_input[:, :, :, 0])
                self.real_loss = mf.l1_loss(self.gen_voice[:, :, :, 1], voice_input[:, :, :, 1])
                self.imag_loss = mf.l1_loss(self.gen_voice[:, :, :, 2], voice_input[:, :, :, 2])
                self.cost = (self. mag_loss + self.real_loss + self.imag_loss) / 3

            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate,
                beta1=0.5,
            )
            self.train_op = self.optimizer.minimize(self.cost)


class UNet(object):
    """
    Magnitude model U-Net
    """
    def __init__(self, input_tensor, variant, data_type, is_training, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            self.variant = variant

            if self.variant == 'unet':
                self.encoder = UNetEncoder(input_tensor, is_training, reuse)
                self.decoder = UNetDecoder(self.encoder.output, self.encoder, data_type, is_training, reuse)
            elif self.variant == 'capsunet':
                self.encoder = CapsUNetEncoder(input_tensor, is_training, reuse)
                self.decoder = CapsUNetDecoder(self.encoder.output, self.encoder, is_training, reuse)

            self.output = mf.tanh(self.decoder.output) / 2 + .5


class UNetEncoder(object):
    """
    The down-convolution side of a convoltional U-Net model.
    """

    def __init__(self, input_tensor, is_training, reuse):

        self.input_tensor = input_tensor
        with tf.variable_scope('encoder'):
            with tf.variable_scope('layer-1'):
                net = mf.conv(self.input_tensor, filters=16, kernel_size=5, stride=(2, 2))
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
    def __init__(self, input_tensor, encoder, data_type, is_training, reuse):
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
                if data_type == 'mag_phase_real_imag':
                    out_shape = 4
                else:
                    out_shape = 2
                net = mf.relu(mf.concat(net, encoder.l1))
                net = mf.deconv(net, filters=out_shape, kernel_size=5, stride=(2, 2))

            self.output = net


class CapsUNetEncoder(object):
    """
    The down-convolutional side of a capsule based U-Net model (based on SegCaps R3 model).
    """

    def __init__(self, input_tensor, is_training, reuse):
        # net = layers.Input(shape=input_tensor)
        net = input_tensor
        with tf.variable_scope('Encoder'):
            with tf.variable_scope('Convolution'):
                # Layer 1: A conventional Conv2D layer
                net = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='same', activation='relu',
                                    name='conv1')(net)
                self.conv1 = net

                # Reshape layer to be 1 capsule x [filters] atoms
                _, H, W, C = net.get_shape()
                net = layers.Reshape((H.value, W.value, 1, C.value))(net)

            # Layer 1: Primary Capsule: Conv cap with routing 1
            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=2, num_atoms=8, strides=2, padding='same',
                                                  routings=1, name='primarycaps')(net)
            self.primary_caps = net

            # Layer 2: Convolutional Capsules
            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=8, strides=2, padding='same',
                                                  routings=3, name='conv_cap_2')(net)
            self.conv_cap_2 = net

            # Layer 3: Convolutional Capsules
            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=16, strides=2, padding='same',
                                                  routings=3, name='conv_cap_3')(net)
            self.conv_cap_3 = net

            # Layer 4: Convolutional Capsules
            net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=32, strides=2, padding='same',
                                                  routings=3, name='conv_cap_4')(net)

            self.output = net


class CapsUNetDecoder(object):
    """
    The up-convolutional side of a capsule based U-Net model.
    """

    def __init__(self, input_tensor, encoder, is_training, reuse):
        net = input_tensor
        with tf.variable_scope('Decoder'):
            # Layer 1 Up: Deconvolutional capsules, skip connection, convolutional capsules
            net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=8, num_atoms=16, upsamp_type='deconv',
                                                    scaling=2, padding='same', routings=3, name='deconv_cap_1')(net)
            self.upcap_1 = net

            net = layers.Concatenate(axis=-2, name='skip_1')([net, encoder.conv_cap_3])

            # Layer 2 Up: Deconvolutional capsules, skip connection, convolutional capsules
            net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=8, upsamp_type='deconv',
                                                    scaling=2, padding='same', routings=3, name='deconv_cap_2')(net)
            self.upcap_2 = net

            net = layers.Concatenate(axis=-2, name='skip_2')([net, encoder.conv_cap_2])

            # Layer 3 Up: Deconvolutional capsules, skip connection
            net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=2, num_atoms=8, upsamp_type='deconv',
                                                    scaling=2, padding='same', routings=3, name='deconv_cap_3')(net)
            self.upcap_3 = net

            net = layers.Concatenate(axis=-2, name='skip_3')([net, encoder.primary_caps])

            # Layer 4 Up: Deconvolutional capsules, skip connection
            net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=16, upsamp_type='deconv',
                                                    scaling=2, padding='same', routings=3, name='deconv_cap_4')(net)
            self.upcap_4 = net

            # Reconstruction - Reshape, skip connection + 3x conventional Conv2D layers
            _, H, W, C, D = net.get_shape()

            net = layers.Reshape((H.value, W.value, D.value))(net)
            net = layers.Concatenate(axis=-1, name='skip_4')([net, encoder.conv1])

            net = layers.Conv2D(filters=64, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_1')(net)

            net = layers.Conv2D(filters=128, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='relu', name='recon_2')(net)

            net = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer='he_normal',
                                activation='sigmoid', name='out_recon')(net)

            self.output = net


class BasicCapsnet(object):

    def __init__(self, mixed_mag, name):
        """
        A basic capsule network operating on magnitude spectrograms.
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


class conv_net(object):
    def __init__(self, mixed_mag, is_training, reuse, name):
        """
        input_tensor: Tensor with shape [batch_size, height, width, channels]
        is_training:  Boolean - should the model be trained on the current input or not
        name:         Model instance name
        """
        with tf.variable_scope(name):
            self.mixed_mag = mixed_mag

            with tf.variable_scope('Convolution'):
                net = mf.relu(mixed_mag)
                net = mf.conv(net, filters=128, kernel_size=5, stride=(1, 1))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.conv1 = net

            with tf.variable_scope('Primary_Caps'):
                net = mf.relu(net)
                net = mf.conv(net, filters=128, kernel_size=5, stride=(1, 1))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.primary_caps = net

            with tf.variable_scope('Seg_Caps'):
                net = mf.relu(net)
                net = mf.conv(net, filters=16, kernel_size=5, stride=(1, 1))
                net = mf.batch_norm(net, is_training=is_training, reuse=reuse)
                self.seg_caps = net

            with tf.variable_scope('Mask'):
                net = mf.relu(net)
                net = mf.conv(mixed_mag, filters=1, kernel_size=5, stride=(1, 1))
                self.voice_mask = net

            self.output = net


class ComplexNumberModel(object):
    """
    Top level object for models working on complex number spectrograms.
    Attributes:
        mixed_spec: Input placeholder for spectrogram of mixed signals (voice plus background noise), with real number
                    in channel 0 and complex number in channel 1. - X
        voice_spec: Input placeholder for magnitude spectrogram of isolated voice signal - Y
        mixed_audio: Input placeholder for waveform audio of mixed signals (voice plus background noise)
        voice_audio: Input placeholder for waveform audio of isolated voice signal
        variant: The type of U-Net model (Normal convolutional or capsule based)
        is_training: Boolean - should the model be trained on the current input or not
        learning_rate: The learning rate the model should be trained with.
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
    """
    Complex number model U-Net
    """
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
                net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=2, strides=2,
                                                      padding='same',
                                                      routings=3)(net)
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=4, num_atoms=4, strides=2,
                                                      padding='same',
                                                      routings=3)(net)
                self.l3 = net

            with tf.variable_scope('layer-4'):
                net = capsule_layers.ConvCapsuleLayer(kernel_size=5, num_capsule=8, num_atoms=8, strides=2,
                                                      padding='same',
                                                      routings=3)(net)

        self.output = net


class ComplexDecoder(object):
    def __init__(self, input_tensor, encoder):
        net = input_tensor
        with tf.variable_scope('decoder'):
            with tf.variable_scope('layer-1'):
                net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=4, upsamp_type='deconv',
                                                        scaling=2, padding='same', routings=3)(net)
                net = layers.Concatenate(axis=-2)([net, encoder.l3])
                self.l1 = net

            with tf.variable_scope('layer-2'):
                net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=4, num_atoms=2, upsamp_type='deconv',
                                                        scaling=2, padding='same', routings=3)(net)
                net = layers.Concatenate(axis=-2)([net, encoder.l2])
                self.l2 = net

            with tf.variable_scope('layer-3'):
                net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=2, upsamp_type='deconv',
                                                        scaling=2, padding='same', routings=3)(net)
                net = layers.Concatenate(axis=-2)([net, encoder.l1])
                self.l3 = net
            with tf.variable_scope('layer-4'):
                net = capsule_layers.DeconvCapsuleLayer(kernel_size=4, num_capsule=1, num_atoms=2, upsamp_type='deconv',
                                                        scaling=1, padding='same', routings=3)(net)

                net = layers.Reshape((encoder.H.value, encoder.W.value, encoder.C.value))(net)

            self.output = net
