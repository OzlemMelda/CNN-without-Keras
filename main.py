from __future__ import print_function
from builtins import range
from six.moves import cPickle as pickle
from sklearn.utils import shuffle
import numpy as np
import os
import platform
import tensorflow as tf
import datetime
import model
import matplotlib.pyplot as plt
import pickle

# Reference
# https://www.tensorflow.org/api_docs/python/tf/compat/v1/train
# https://www.wouterbulten.nl/blog/tech/data-augmentation-using-tensorflow-data-dataset/

# Constant to control how often we print when training models.
print_every = 80

# Data Load Helpers
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_cifar_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_cifar10(root):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(root, 'data_batch_%d' % (b, ))
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_cifar_batch(os.path.join(root, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

# Data Augmentation Helpers
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32))
        # Return a random crop
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

# Prepare Input Data
def prepare_cifar10(num_training=72000, num_validation=2000, num_test=10000):

    cifar10_dir = 'cifar10_data/cifar10_data/'
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)

    # Normalize the data: subtract the mean pixel and divide by std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel) / std_pixel
    X_test = (X_test - mean_pixel) / std_pixel

    # Data Augmentation
    number_of_rows = X_train.shape[0]

    # Choose and Add augmentations
    augmentations = [[flip], [rotate], [color], [zoom], [flip, color], [rotate, zoom]]

    for aug in augmentations:

        random_indices = np.random.choice(number_of_rows,
                                          size=4000,
                                          replace=False)
        data = X_train[random_indices, :, :, :].astype(np.float32)
        y_train_temp = y_train[random_indices]

        dataset = tf.data.Dataset.from_tensor_slices(data)

        for f in aug:
            dataset = dataset.map(lambda x: f(x),
                                  num_parallel_calls=4)
        dataset = dataset.map(lambda x: tf.clip_by_value(x, 0, 1))

        output = np.zeros((4000, 32, 32, 3))
        row = 0
        for images in dataset.repeat(1).batch(4000):
            output[:, :, row * 32:(row + 1) * 32] = np.stack(images.numpy())
            row += 1

        X_train = np.concatenate((X_train, output), axis=0)
        y_train = np.concatenate((y_train, y_train_temp), axis=0)

    # Subsample The Data
    X_train, y_train = shuffle(X_train, y_train, random_state=0)

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = prepare_cifar10()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape, y_train.dtype)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Dump Test Data for Evaluation
np.save('X_test', X_test)
np.save('y_test', y_test)

# Prepare Batches
class Dataset(object):
    def __init__(self, X, y, batch_size, shuffle=False):
        assert X.shape[0] == y.shape[0], 'Got different numbers of data and labels'
        self.X, self.y = X, y
        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i + B], self.y[i:i + B]) for i in range(0, N, B))


train_dset = Dataset(X_train, y_train, batch_size=256, shuffle=True)


# Training Model
def training(model_fn, learning_rate, num_epochs):
    k = 0

    # for early stopping
    best_loss = 1000000
    last_improvement = 0
    require_improvement = 20

    # loss and accuracy history
    t_losses = []
    val_losses = []
    t_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print('Epoch Number %d' % epoch)
        avg_t_loss = 0
        avg_val_loss = 0
        avg_t_acc = 0
        avg_val_acc = 0
        count_t = 0
        count_v = 0
        print(datetime.datetime.now())
        for t, (x_np, y_np) in enumerate(train_dset):
            # Run the graph on a batch of training data.
            t_loss, t_accuracy = model_fn.bp(x_np, y_np, learning_rate)

            avg_t_loss += t_loss.numpy()
            avg_t_acc += t_accuracy
            count_t = count_t + 1
            # Periodically print the loss and check accuracy on the train and val set
            if t % print_every == 0:
                val_loss, val_accuracy = model_fn.check_accuracy(X_val, y_val)

                avg_val_loss += val_loss.numpy()
                avg_val_acc += val_accuracy
                count_v = count_v + 1
                print('Iteration %d, Training Loss: %.4f, Training Accuracy: %.4f' % (t, t_loss, t_accuracy))
                print('Iteration %d, Val Loss: %.4f, Val Accuracy: (%.2f%%)' % (t, val_loss, 100*val_accuracy))

        avg_t_loss = avg_t_loss / count_t
        avg_val_loss = avg_val_loss / count_v
        avg_t_acc = avg_t_acc / count_t
        avg_val_acc = avg_val_acc / count_v

        t_losses.append(avg_t_loss)
        val_losses.append(avg_val_loss)
        t_accuracies.append(avg_t_acc)
        val_accuracies.append(avg_val_acc)

        # Early stopping based on the validation set/ max_steps_without_decrease of the loss value : require_improvement
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            last_improvement = 0
        else:
            last_improvement += 1

        # Dump model at the beginning, at the middle or at the end of training
        if (epoch == 1) | (epoch == 19) | (epoch == 99) | (last_improvement == require_improvement):
            if (epoch == 1) | (epoch == 19) | (epoch == 99):
                pkl_filename = "model_" + str(epoch) + '.pk'
            if last_improvement == require_improvement:
                pkl_filename = "model_" + str(last_improvement) + '.pk'
            with open(pkl_filename, 'wb') as file:
                pickle.dump(conv, file)

        if last_improvement > require_improvement:
            print("No improvement found during the last iterations, early stopping optimization.")
            # Break out from the loop
            break
        print('last_improvement: %d', last_improvement)
        k = k+1

    return t_losses, val_losses, t_accuracies, val_accuracies


conv = model.three_layer_convnet()
learning_rate = 0.02
num_epochs = 100
t_losses, val_losses, t_accuracies, val_accuracies = training(conv, learning_rate, num_epochs)
