import numpy as np
import tensorflow as tf
import gpflow
import os

from gpflow import transforms
from gpflow.kullback_leiblers import gauss_kl
from gpflow.params import Parameter
from gpflow.test_util import notebook_niter, is_continuous_integration
from gpflow.decors import params_as_tensors, autoflow
import gpflow.training.monitor as mon
from gpflow.models import GPModel
import sys

from stn import spatial_transformer_network as transformer


from scipy.cluster.vq import kmeans2


def rotate(x, mins, maxes, image_shape=[28, 28, 1]):
    angle = tf.random_uniform(shape=(), minval=mins, maxval=maxes,
                             dtype=tf.float32)
    # Rotation matrix + zero bias term
    theta = [tf.cos(angle), -tf.sin(angle), 0, tf.sin(angle), tf.cos(angle), 0]
    B, H, W, C = x.shape
    # define loc net weight and bias
    loc_in = H*W*C
    loc_out = 6
    W_loc = tf.constant(tf.zeros([loc_in, loc_out]), name='W_loc', trainable=False)
    b_loc = tf.constant(value=theta, name='b_loc', trainable=False)
    
    # tie everything together
    fc_loc = tf.matmul(tf.zeros([B, loc_in]), W_loc) + b_loc
    h_trans = transformer(x, fc_loc)
    return h_trans

def c4_rotate(x, i):
    angle = 0.5* np.pi * tf.cast(i, tf.float32)
     # Rotation matrix + zero bias term
    h_trans = tf.image.rot90(x,k=i,name=None)
    return h_trans

def gaussian(x, radius):
    perturbation = tf.random.normal(x.get_shape(), stddev=radius)
    return x + perturbation

def uniform(x, radius):
    perturbation = tf.random.uniform(x.get_shape(), stddev=radius)
    return x + perturbation

# TODO: add STN transformations