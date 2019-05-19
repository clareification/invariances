# Visualize conv net features
import utils
import numpy as np
import tensorflow as tf
import os
import variational_model as vm
import pickle as pkl
from keras.utils.np_utils import to_categorical
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import matplotlib.pyplot as plt
from PIL import Image
tf.logging.set_verbosity(tf.logging.ERROR)
PLOT_DIR = './noav4x_plots'


def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))

    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            img = np.reshape(img, (img.shape[0], img.shape[1]))# put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation=None, cmap='Greys')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

filename='rotatedp4.pickle'
with open(filename, 'rb') as f:
    d = pkl.load(f)
    unambiguous_X = d['train_ims'][:60000]
    unambiguous_Y = d['train_labels'][:60000]
    es = d['test_ims']
    ls = d['test_labels']

float_type=np.float64


class Mnist:
    input_dim = 784
    Nclasses = 10
    X =  unambiguous_X
    Y = to_categorical(unambiguous_Y, 10)
    Xtest = es
    Ytest = to_categorical(ls, 10)
for i in range(1,6):
    tf.reset_default_graph()
    dvm = vm.DeepVM(60, 10, averaging=None, ckpt_dir=None, train_features=True, equivariant=None)
    with tf.variable_scope('deepVM', reuse=tf.AUTO_REUSE): 
        features = dvm.feature_map(Mnist.Xtest[0])
    pred = dvm.predict(features)
    print(tf.all_variables())
     # print(tf.get_collection(tf.GraphKeys.VARIABLES), tf.get_variable('conv2d_1/kernel:0'))
    # lb = dvm.lower_bound(features, Mnist.Ytest)
    print(tf.global_variables())
    sv = tf.train.Supervisor(logdir='tdlfashion/noav4xfilters/' + str(i))
    latest_ckpt = tf.train.latest_checkpoint('tdlfashion/noav4xfilters/' + str(i))
    print_tensors_in_checkpoint_file(latest_ckpt, all_tensors=True, tensor_name='')
    with sv.managed_session() as s:
        conv_imgs = s.run(tf.get_collection('conv_output1'))
        p = s.run(pred)
        weights = s.run([v for v in tf.global_variables() if v.name == 'deepVM/conv2d/kernel:0'])
    print(conv_imgs)
    # conv_imgb = conv_imgs[0]
    print(weights[0].shape)
    print(weights[0][:,:, :])
    plot_conv_weights(weights[0], str(i))
    # print(conv_imgb)
    conv_imgb = conv_imgs[0]
    print(conv_imgb.shape)
    # print(p, Mnist.Ytest[0])
    for i in range(8):
        conv_img = conv_imgb[:, :, :, i].reshape(28, 28)
        # print(conv_img)
        # print(conv_img.shape)
        conv_img = 256 * conv_img
        w_min = np.min(conv_imgb)
        w_max = np.max(conv_imgb)
        a_max = np.argmax(conv_imgb)
        print("Max: ", w_max, w_min) #,  conv_imgb[a_max/28, a_max%28])
        img = Image.fromarray(conv_img/w_max, 'F')
        img.convert('RGB').save('test' + str(i) + '.png', 'PNG')

