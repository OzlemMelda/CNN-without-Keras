from __future__ import print_function
import numpy as np
import tensorflow as tf


class three_layer_convnet():

    def __init__(self):

        # Weights initialization
        self.conv_w1 = tf.Variable(self.create_matrix_with_kaiming_normal((3, 3, 3, 32)))
        self.conv_w2 = tf.Variable(self.create_matrix_with_kaiming_normal((3, 3, 32, 64)))
        self.conv_w3 = tf.Variable(self.create_matrix_with_kaiming_normal((3, 3, 64, 128)))
        # self.conv_wr = tf.Variable(self.create_matrix_with_kaiming_normal((3, 3, 64, 64)))
        self.fc_w1 = tf.Variable(self.create_matrix_with_kaiming_normal((2048, 1024)))  # 4096
        self.fc_w2 = tf.Variable(self.create_matrix_with_kaiming_normal((1024, 512)))
        self.out_w = tf.Variable(self.create_matrix_with_kaiming_normal((512, 10)))

        # Bias initialization
        self.conv_b1 = tf.Variable(tf.zeros((32,)))
        self.conv_b2 = tf.Variable(tf.zeros((64,)))
        self.conv_b3 = tf.Variable(tf.zeros((128,)))
        # self.conv_br = tf.Variable(tf.zeros((64,)))
        self.fc_b1 = tf.Variable(tf.zeros((1024,)))
        self.fc_b2 = tf.Variable(tf.zeros((512,)))
        self.out_b = tf.Variable(tf.zeros((10,)))

        self.params = [self.conv_w1,
                       self.conv_b1,
                       self.conv_w2,
                       self.conv_b2,
                       self.conv_w3,
                       self.conv_b3,
                       # self.conv_wr,
                       # self.conv_br,
                       self.fc_w1,
                       self.fc_w2,
                       self.fc_b1,
                       self.fc_b2,
                       self.out_w,
                       self.out_b]

        self.h1_history_x = []
        self.h2_history_x = []
        self.h3_history_x = []
        self.h4_history_x = []
        self.h5_history_x = []
        self.h_history_y = []

    @staticmethod
    def flatten(x):
        N = tf.shape(x)[0]
        return tf.reshape(x, (N, -1))

    @staticmethod
    def create_matrix_with_kaiming_normal(shape):
        """ Initialize the parameters """
        if len(shape) == 2:
            fan_in, fan_out = shape[0], shape[1]
        elif len(shape) == 4:
            fan_in, fan_out = np.prod(shape[:3]), shape[3]
        return tf.keras.backend.random_normal(shape) * np.sqrt(2.0 / fan_in)

    def conv2d(self, x, W, b):
        """ Convolution Layer """
        x = tf.nn.conv2d(x,
                          W,
                          strides=1,
                          padding='SAME')
        x = tf.compat.v1.layers.batch_normalization(x, trainable=True)
        x = tf.nn.relu(tf.nn.bias_add(x, b))
        return x

    def fcl(self, x, W, b):
        """ Fully Connected Layer """
        x = tf.nn.bias_add(tf.matmul(x, W), b)
        x = tf.nn.relu(x)
        return x

    """def resnet(self, x):
        # Residual Block
        x_out = self.conv2d(x, self.conv_wr, self.conv_br)
        res = tf.add(x_out, x)
        res = tf.nn.relu(res)
        return res"""

    def max_pool(self, x, k=2, s=2):
        """ Pooling """
        return tf.nn.max_pool(x, ksize=k, strides=s, padding='SAME')

    def fp(self, x, y, type):
        """ Forward Propagation """
        h1 = self.conv2d(x, self.conv_w1, self.conv_b1)
        h1 = self.max_pool(h1)

        h2 = self.conv2d(h1, self.conv_w2, self.conv_b2)
        h2 = self.max_pool(h2)

        h3 = self.conv2d(h2, self.conv_w3, self.conv_b3)
        h3 = self.max_pool(h3)

        # resnet
        # h3 = self.resnet(h2)

        h4 = self.flatten(h3)
        h4 = self.fcl(h4, self.fc_w1, self.fc_b1)

        h5 = self.fcl(h4, self.fc_w2, self.fc_b2)

        # Output, class prediction
        scores = tf.add(tf.matmul(h5, self.out_w), self.out_b)

        if type == 'check':

            self.h1_history_x.append(h1.numpy().copy())
            self.h2_history_x.append(h2.numpy().copy())
            self.h3_history_x.append(h3.numpy().copy())
            self.h4_history_x.append(h4.numpy().copy())
            self.h5_history_x.append(h5.numpy().copy())
            self.h_history_y.append(y.copy())

        return scores

    def bp(self, x, y, learning_rate):
        """ Backward Propagation """
        with tf.GradientTape() as tape:
            scores = self.fp(x, y, type="train")  # Forward pass of the model
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
            total_loss = tf.reduce_mean(loss)
            grad_params = tape.gradient(total_loss, self.params)

            scores_np = scores.numpy()
            y_pred = scores_np.argmax(axis=1)
            num_samples = x.shape[0]
            num_correct = (y_pred == y).sum()
            accuracy = float(num_correct) / num_samples

            optimizer = tf.compat.v1.train.AdagradOptimizer(
                learning_rate=learning_rate, initial_accumulator_value=0.1, use_locking=False,
                name='Adagrad'
            )

            """optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate=learning_rate, use_locking=False, name='GradientDescent'
            )"""

            """optimizer = tf.compat.v1.train.RMSPropOptimizer(
                learning_rate, decay=0.99, momentum=0.2, epsilon=1e-10, use_locking=False,
                centered=False, name='RMSProp'
            )"""

            optimizer.apply_gradients(zip(grad_params, self.params))

            return total_loss, accuracy

    def check_accuracy(self, x, y):

        scores = self.fp(x, y=None, type="validation")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
        total_loss = tf.reduce_mean(loss)

        scores_np = scores.numpy()
        y_pred = scores_np.argmax(axis=1)
        num_samples = x.shape[0]
        num_correct = (y_pred == y).sum()
        accuracy = float(num_correct) / num_samples

        return total_loss, accuracy