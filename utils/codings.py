import tensorflow as tf

from tensorflow.contrib import layers
from planar import Planar
from tensorflow.contrib.framework import arg_scope
from activations import lrelu,leaky_rectify
import tensorlayer as tl
from tensorlayer.layers import *

def planar_projector(e, k, latent_dim):
    z = e
    log_det= 0
    for i in range(1, k+1):
        z, l = Planar(scope='planar_'+str(i), activation=tf.nn.tanh)(z)
    	log_det += l
    return z, log_det


def fc_noise_projector(e, k, latent_dim, activation=tf.tanh):
    z = e
    for i in range(1, k+1):
        z = layers.fully_connected(z, 64, scope='fc_'+str(i),
	                           activation_fn=activation,
	                           #normalizer_fn=tf.contrib.layers.batch_norm
	                           )
    z = layers.fully_connected(z, latent_dim, activation_fn=None,
                               scope='fc-final')
    return z

def fc_mnist_encoder(x, latent_dim, activation=None):
    e = layers.fully_connected(x, 500, scope='fc-01')
    e = layers.fully_connected(e, 500, scope='fc-02')
    e = layers.fully_connected(e, 200, scope='fc-03')
    e = layers.fully_connected(e, 2 * latent_dim, activation_fn=activation,
                               scope='fc-final')

    return e


def fc_mnist_decoder(z, z_size=None, activation=tf.nn.sigmoid):
    x = layers.fully_connected(z, 200, scope='fc-01')
    x = layers.fully_connected(x, 500, scope='fc-02')
    x = layers.fully_connected(x, 500, scope='fc-03')
    x = layers.fully_connected(x, 28 * 28, activation_fn=activation,
                               scope='fc-final')

    return x, None

def conv_mnist_encoder(x, latent_dim, activation=None):
    e = tf.reshape(x, [-1, 28, 28, 1])
    e = layers.conv2d(e, 32, 5, stride=2, scope='conv-01')
    e = layers.conv2d(e, 64, 5, stride=2, scope='conv-02')
    e = layers.conv2d(e, 128, 3, stride=2, scope='conv-03')
    e = layers.flatten(e)
    e = layers.fully_connected(e, 500, scope='fc-01')
    e = layers.fully_connected(e, 200, scope='fc-02')
    e = layers.fully_connected(e, 2 * latent_dim, activation_fn=None,
                               scope='conv-final')

    return e


def conv_mnist_decoder(z, z_size=None, activation=tf.nn.sigmoid):
    x = tf.expand_dims(z, 1)
    x = tf.expand_dims(x, 1)
    x = layers.conv2d_transpose(x, 128, 3, padding='VALID', scope='conv-transpose-01')
    x = layers.conv2d_transpose(x, 64, 5, padding='VALID', scope='conv-transpose-02')
    x = layers.conv2d_transpose(x, 32, 5, stride=2, scope='conv-transpose-03')
    x = layers.conv2d_transpose(x, 1, 5, stride=2, activation_fn=activation,
                                scope='conv-transpose-final')
    x = layers.flatten(x)

    return x, None

def conv_celebA_decoder(z, is_train):

    image_size = 64
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16) # 32,16,8,4
    gf_dim = 64
    c_dim = 3

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    net_in = InputLayer(z, name='g/in')
    net_h0 = DenseLayer(net_in, n_units=gf_dim*4*s8*s8, W_init=w_init,
            act = tf.identity, name='g/h0/lin')
    # net_h0.outputs._shape = (b_size,256*8*8)
    net_h0 = ReshapeLayer(net_h0, shape=[-1, s8, s8, gf_dim*4], name='g/h0/reshape')
    # net_h0.outputs._shape = (b_size,8,8,256)
    net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
            gamma_init=gamma_init, name='g/h0/batch_norm')

    # upsampling
    net_h1 = DeConv2d(net_h0, gf_dim*4, (5, 5), out_size=(s4, s4), strides=(2, 2),
            padding='SAME', act=None, W_init=w_init, name='g/h1/decon2d')
    net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
            gamma_init=gamma_init, name='g/h1/batch_norm')
    # net_h1.outputs._shape = (b_size,16,16,256)

    net_h2 = DeConv2d(net_h1, gf_dim*2, (5, 5), out_size=(s2, s2), strides=(2, 2),
            padding='SAME', act=None, W_init=w_init, name='g/h2/decon2d')
    net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
            gamma_init=gamma_init, name='g/h2/batch_norm')
    # net_h2.outputs._shape = (b_size,32,32,128)

    net_h3 = DeConv2d(net_h2, gf_dim//2, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
            padding='SAME', act=None, W_init=w_init, name='g/h3/decon2d')
    net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
            gamma_init=gamma_init, name='g/h3/batch_norm')
    # net_h3.outputs._shape = (b_size,64,64,32)

    # no BN on last deconv
    net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(image_size, image_size), strides=(1, 1),
            padding='SAME', act=None, W_init=w_init, name='g/h4/decon2d')
    # net_h4.outputs._shape = (b_size,64,64,3)

    logits = net_h4.outputs
    net_h4.outputs = tf.nn.tanh(net_h4.outputs)

    output = layers.flatten(net_h4.outputs)
    return output, net_h4


def conv_celebA_enhanced_decoder(inputs, is_train = True, reuse = False):

    image_size = 64
    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16) # 32,16,8,4
    gf_dim = 32
    c_dim = 3
    batch_size = 64

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    with tf.variable_scope("generator", reuse = reuse):
        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs, name='g/in')
        net_h0 = DenseLayer(net_in, n_units=gf_dim*8*s16*s16, W_init=w_init,
                act = tf.identity, name='g/h0/lin')

        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*8], name='g/h0/reshape')

        net_h0 = BatchNormLayer(net_h0, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                gamma_init=gamma_init, name='g/h0/batch_norm')

        # upsampling
        net_h1 = UpSampling2dLayer(net_h0, size=[8, 8], is_scale=False, method=1,
                                    align_corners=False, name='g/h1/upsample2d')
        net_h1 = Conv2d(net_h1, gf_dim*4, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='g/h1/conv2d')

        net_h1 = BatchNormLayer(net_h1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                gamma_init=gamma_init, name='g/h1/batch_norm')
        # net_h1.outputs._shape = (b_size,8,8,128)

        net_h2 = UpSampling2dLayer(net_h1, size=[16, 16], is_scale=False, method=1,
                                    align_corners=False, name='g/h2/upsample2d')
        net_h2 = Conv2d(net_h2, gf_dim*2, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='g/h2/conv2d')

        net_h2 = BatchNormLayer(net_h2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                gamma_init=gamma_init, name='g/h2/batch_norm')
        # net_h2.outputs._shape = (b_size,16,16,64)

        net_h3 = UpSampling2dLayer(net_h2, size=[32, 32], is_scale=False, method=1,
                                    align_corners=False, name='g/h3/upsample2d')
        net_h3 = Conv2d(net_h3, gf_dim, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='g/h3/conv2d')

        net_h3 = BatchNormLayer(net_h3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                gamma_init=gamma_init, name='g/h3/batch_norm')

        net_h4 = UpSampling2dLayer(net_h3, size=[64, 64], is_scale=False, method=1,
                                    align_corners=False, name='g/h4/upsample2d')
        net_h4 = Conv2d(net_h4, c_dim, (3, 3), (1, 1), padding='SAME', W_init=w_init, name='g/h4/conv2d')

        net_h4.outputs = tf.nn.tanh(net_h4.outputs)

        output = layers.flatten(net_h4.outputs)
    return output, net_h4

def conv_anime_encoder(x, latent_dim, activation=None):
    is_train = True
    z_dim = latent_dim  # 512
    ef_dim = 64  # encoder filter number

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    input = tf.reshape(x, [-1, 64, 64, 3])
    net_in = InputLayer(input, name='en/in')  # (b_size,64,64,3)
    net_h0 = Conv2d(net_in, ef_dim, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='en/h0/conv2d')
    net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu,
                            is_train=is_train, gamma_init=gamma_init, name='en/h0/batch_norm')

    net_h1 = Conv2d(net_h0, ef_dim * 2, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='en/h1/conv2d')
    net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu,
                            is_train=is_train, gamma_init=gamma_init, name='en/h1/batch_norm')

    net_h2 = Conv2d(net_h1, ef_dim * 4, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='en/h2/conv2d')
    net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu,
                            is_train=is_train, gamma_init=gamma_init, name='en/h2/batch_norm')

    net_h3 = Conv2d(net_h2, ef_dim * 8, (5, 5), (2, 2), act=None,
                    padding='SAME', W_init=w_init, name='en/h3/conv2d')
    net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu,
                            is_train=is_train, gamma_init=gamma_init, name='en/h3/batch_norm')


    net_h4 = FlattenLayer(net_h3, name='en/h4/flatten')

    net_out = DenseLayer(net_h4, n_units=z_dim*2, act=tf.identity,
                         W_init=w_init, name='en/h3/lin_sigmoid')
    net_out = BatchNormLayer(net_out, act=tf.identity,
                             is_train=is_train, gamma_init=gamma_init, name='en/out1/batch_norm')

    output = net_out.outputs

    return output


def conv_anime_decoder(z, observation_dim, activation=tf.tanh):

    is_train = True
    image_size = 64
    s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)  # 32,16,8,4
    gf_dim = 64
    c_dim = 3

    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)

    net_in = InputLayer(z, name='g/in')
    net_h0 = DenseLayer(net_in, n_units=gf_dim * 4 * s8 * s8, W_init=w_init,
                        act=tf.identity, name='g/h0/lin')
    # net_h0.outputs._shape = (b_size,256*8*8)
    net_h0 = ReshapeLayer(net_h0, shape=[-1, s8, s8, gf_dim * 4], name='g/h0/reshape')
    # net_h0.outputs._shape = (b_size,8,8,256)
    net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                            gamma_init=gamma_init, name='g/h0/batch_norm')

    # upsampling
    net_h1 = DeConv2d(net_h0, gf_dim * 4, (5, 5), out_size=(s4, s4), strides=(2, 2),
                      padding='SAME', act=None, W_init=w_init, name='g/h1/decon2d')
    net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                            gamma_init=gamma_init, name='g/h1/batch_norm')
    # net_h1.outputs._shape = (b_size,16,16,256)

    net_h2 = DeConv2d(net_h1, gf_dim * 2, (5, 5), out_size=(s2, s2), strides=(2, 2),
                      padding='SAME', act=None, W_init=w_init, name='g/h2/decon2d')
    net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                            gamma_init=gamma_init, name='g/h2/batch_norm')
    # net_h2.outputs._shape = (b_size,32,32,128)

    net_h3 = DeConv2d(net_h2, gf_dim // 2, (5, 5), out_size=(image_size, image_size), strides=(2, 2),
                      padding='SAME', act=None, W_init=w_init, name='g/h3/decon2d')
    net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                            gamma_init=gamma_init, name='g/h3/batch_norm')
    # net_h3.outputs._shape = (b_size,64,64,32)

    # no BN on last deconv
    net_h4 = DeConv2d(net_h3, c_dim, (5, 5), out_size=(image_size, image_size), strides=(1, 1),
                      padding='SAME', act=None, W_init=w_init, name='g/h4/decon2d')
    # net_h4.outputs._shape = (b_size,64,64,3)

    logits = net_h4.outputs
    net_h4.outputs = tf.nn.tanh(net_h4.outputs)

    output = layers.flatten(net_h4.outputs)

    return output, None
