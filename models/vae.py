import re
import numpy as np
import tensorflow as tf


class VAE(object):

    def __init__(self, latent_dim, batch_size, encoder, decoder,
                 obs_dim=784,
                 learning_rate=1e-3,
                 optimizer=tf.train.RMSPropOptimizer,
                 obs_distrib="Bernoulli",
                 obs_std=0.1,
                 ):

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._obs_dim = obs_dim
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._obs_distrib = obs_distrib
        self._obs_std = obs_std
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('vae'):
            # placeholder for MNIST inputs
            self.x = tf.placeholder(tf.float32, shape=[None, self._obs_dim])

            # encode inputs (map to parameterization of diagonal Gaussian)
            with tf.variable_scope('encoder'):
                self.encoded = self._encode(self.x, self._latent_dim)

            with tf.variable_scope('sampling'):
                # extract mean and (diagonal) log variance of latent variable
                self.mean = self.encoded[:, :self._latent_dim]
                self.logvar = self.encoded[:, self._latent_dim:]
                # also calculate standard deviation for practical use
                self.stddev = tf.sqrt(tf.exp(self.logvar))

                # sample from latent space
                epsilon = tf.random_normal([self._batch_size, self._latent_dim])
                self.z = self.mean + self.stddev * epsilon

            # decode batch
            with tf.variable_scope('decoder'):
		print(self._encode)		
		print(self._decode)
                self.decoded, _ = self._decode(self.z)

            with tf.variable_scope('loss'):
                # calculate KL divergence between approximate posterior q and prior p
                with tf.variable_scope('kl-divergence'):
                    kl = self._kl_diagnormal_stdnormal(self.mean, self.logvar)

                # calculate reconstruction error between decoded sample
                # and original input batch
                if self._obs_distrib == 'Bernoulli':
                    with tf.variable_scope('bernoulli'):
                        log_like = self._bernoulli_log_likelihood(self.x, self.decoded)
                else:
                    with tf.variable_scope('gaussian'):
                        log_like = self._gaussian_log_likelihood(self.x, self.decoded, self._obs_std)

                self._loss = (kl + log_like) / self._batch_size
                self._reconstruction_loss = log_like/self._batch_size

            with tf.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)
            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            # start tensorflow session
            self._sesh = tf.Session()
            init = tf.global_variables_initializer()
            self._sesh.run(init)

    def _kl_diagnormal_stdnormal(self, mu, log_var):

        var = tf.exp(log_var)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        return kl

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean)) / (2 * tf.square(std)) + tf.log(std)
        return se


    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, eps=1e-8):

        log_like = -tf.reduce_sum(targets * tf.log(outputs + eps)
                                  + (1. - targets) * tf.log((1. - outputs) + eps))
        return log_like

    def update(self, x):

        _, loss = self._sesh.run([self._train, self._loss],
                                 feed_dict={self.x: x})
        return loss

    def x2z(self, x):

        mean = self._sesh.run([self.mean], feed_dict={self.x: x})
        return np.asarray(mean).reshape(-1, self._latent_dim)

    def z2x(self, z, mnist=True):

        x = self._sesh.run([self.decoded],
                           feed_dict={self.z: z})
        # need to reshape since our network processes batches of 1-D 28 * 28 arrays
        if mnist:
            x = np.array(x)[:, 0, :].reshape(28, 28)
        return x

    def save_generator(self, path, prefix="is/generator"):
        variables = tf.trainable_variables()
        var_dict = {}
        for v in variables:
            if "decoder" in v.name:
                name = prefix+re.sub("vae/decoder", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)
        saver = tf.train.Saver(var_dict)
        saver.save(self._sesh, path)
