import tensorflow as tf

# Tensorflow Function
def get_jacobian(y, x):
    """

    :param y: tensor  MxD dimensions
    :param x: tensor MxD dimensions
    :return: a list of jacobian matrix in tensor format
    """

    y_dim = y.get_shape().as_list()[1]
    x_dim = x.get_shape().as_list()[1]
    assert y_dim == x_dim, "Jacobian Matrix Error: dimension does not match"

    y_list = tf.unstack(y, axis=1)

    jacobian_matrix = []

    for yd in y_list:
        jacobian_matrix.append(tf.gradients(yd, x))

    jacobians = tf.reshape(tf.concat(jacobian_matrix, axis=2), [-1, y_dim, x_dim])

    return jacobians

# Tensorflow Function
def get_determinant(jacobians):
    """

    :param jacobians: a list of jacobian matrix in tensor format
    :return: a list of determinants in tensor format
    """
    determinants = tf.abs(tf.matrix_determinant(jacobians))
    return determinants

# Tensorflow Function
def get_determinant2(jacobians):
    """

    :param jacobians: a list of jacobian matrix in tensor format
    :return: a list of determinants in tensor format
    """
    determinants = tf.matrix_determinant(jacobians)
    sign = tf.sign(determinants)
    abs_determiants = tf.abs(determinants)
    return abs_determiants, sign

# Tensorflow Function
def get_inverse_det_weights(determinants):
    """

    :param determinants: a list of determinants in tensor format
    :return: a list of weights in tensor format
    """
    inverse_det = 1/determinants
    partition = tf.reduce_sum(inverse_det)/tf.cast(tf.shape(determinants), tf.float32)
    weights = inverse_det/partition

    # partition = tf.reduce_sum(determinants)
    # weights = determinants/partition

    return weights



# TEST CODE
# x = tf.get_variable("x", [10000, 10])
# y = x + 100
# jacobian = get_jacobian(y,x)
# print(jacobian)
# determinant = get_determinant(y,x)
# print(determinant)
# weights = get_inverse_det_weights(y,x)
# print(weights)


