import tensorflow as tf
from utils.jacobian import get_determinant, get_jacobian

class Planar(object):
    """Planar Normalization Flow"""

    def __init__(self, scope="planar", activation=tf.identity):
        self.scope = scope
        self.activation = activation

    def __call__(self, input):
        with tf.name_scope(self.scope):
            self.output_dim = input.get_shape()[1].value
            while True:
                try:
                    z = input
                    uw = tf.matmul(tf.transpose(self.u), self.w)
                    muw = -1 + tf.log(1 + tf.exp(uw))
                    u_hat = self.u + tf.matmul(self.w / tf.reduce_sum(self.w ** 2), (muw - uw))
                    zwb = tf.matmul(z, self.w) + self.b
                    f_z = z + tf.matmul(tf.nn.tanh(zwb), tf.transpose(u_hat))

                    psi = tf.matmul(1 - tf.nn.tanh(zwb) ** 2, tf.transpose(self.w))
                    psi_u = tf.matmul(psi, u_hat)
                    logdet = tf.reshape(-tf.log(tf.abs(1 + psi_u)), [-1])

                    # determinant = get_determinant(get_jacobian(f_z, z))
                    # logdet = -tf.log(determinant)

                    return f_z, logdet

                except Exception:
                    self.w = self.weight_variable([self.output_dim, 1], 'w')
                    self.u = self.weight_variable([self.output_dim, 1], 'u')
                    self.b = self.bias_variable()

    # Weight constructing function
    @staticmethod
    def weight_variable(shape, name="w"):
        initial = tf.truncated_normal(shape, stddev=0.005)
        return tf.Variable(initial, name=name)

    # Bias constructing function
    @staticmethod
    def bias_variable():
        initial = tf.constant(0.)
        return tf.Variable(initial, name="b")


# # TEST CODE
# planar = Planar()
# x = tf.placeholder(tf.float32, shape=[100, 2])
# z, log_det = planar(x)
# print(log_det.get_shape())
# print("Graph Generated")