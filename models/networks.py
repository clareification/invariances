import gin.tf
import numpy as np
import tensorflow as tf
import os
import model_comparison.models.transformations as iv
import tensorflow.contrib.slim as slim
from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d, gconv2d_util
from tensorflow.keras import layers




def cnn_fn(x, output_dim, trainable=True, group=None, mnist=True, num_filters=64):
    """t
    Adapted from https://www.tensorflow.org/tutorials/layers
    """
    if not mnist:
        input_shape = [32, 32, 3]
    else:
        input_shape = [28, 28, 1]
    input_shape = x.shape[1:4]
    
    conv1 = tf.layers.conv2d(
          inputs=x,
          filters=num_filters,
          kernel_size=[5, 5],
          padding="same",
          activation=None,
          trainable=trainable)
    tf.add_to_collection('conv_output1', conv1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # pool1 = layers.Dropout(0.25)(pool1)# pool1 = tf.Print(pool1, [pool1], "Here's pooling: ")
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=num_filters,
          kernel_size=[5, 5],
          padding="same",
          activation=None,
          trainable=trainable)
    tf.add_to_collection('conv_output2', conv2)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
   
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 8])
    pool2_flat = tf.layers.flatten(pool2)
    u = pool2_flat 
    u = tf.layers.dense(inputs=pool2_flat, units=60, activation=tf.nn.relu, trainable=trainable)
    u = tf.layers.dense(inputs = u, units = output_dim, activation=None, trainable=trainable)
    return u



@gin.configurable
def cifar_fn(x, output_dim=10, trainable=True, group=None, mnist=True):
    """t
    Adapted from https://www.tensorflow.org/tutorials/layers
    """
    if not mnist:
        input_shape = [32, 32, 3]
    else:
        input_shape = [28, 28, 1]
    
    conv1 = tf.layers.conv2d(
          inputs=tf.reshape(x, [-1] + input_shape),
          filters=16,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          trainable=trainable)
    tf.add_to_collection('conv_output1', conv1)
    conv1 = tf.layers.conv2d(
          inputs=conv1,
          filters=16,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          trainable=trainable)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    pool1 = layers.Dropout(0.25)(pool1)
    # pool1 = tf.Print(pool1, [pool1], "Here's pooling: ")
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=32,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          trainable=trainable)
    tf.add_to_collection('conv_output2', conv2)
    conv2 = tf.layers.conv2d(
          inputs=conv2,
          filters=32,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu,
          trainable=trainable)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu,
          trainable=trainable)
    
    conv3 = tf.layers.conv2d(
          inputs=conv3,
          filters=64,
          kernel_size=[7, 7],
          padding="same",
          activation=tf.nn.relu,
          trainable=trainable)
    tf.add_to_collection('conv_output3', conv3)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    # pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 8])
    
    pool2_flat = tf.layers.flatten(pool3)
    u = pool2_flat
    # u = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu, trainable=trainable)
    u = tf.layers.dense(inputs=u, units=output_dim, activation=tf.nn.relu, trainable=trainable)
    
    return u

@gin.configurable
def eq_cifar_fn(x, output_dim=10, trainable=True):
        gconv_indices, gconv_shape_info, w_shape = gconv2d_util(h_input='Z2', h_output='C4', in_channels=3, out_channels=8, ksize=3)
        w = tf.get_variable('w1', shape=w_shape)
        
        conv1 = gconv2d(input=x, filter=w,    strides=[1,2,2,1], padding='SAME', gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
        tf.add_to_collection('conv_output1', conv1)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
        gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
                h_input='C4', h_output='C4', in_channels=16, out_channels=32, ksize=5)
        w = tf.get_variable('w2', shape=w_shape)
        conv2 = gconv2d(input=conv1, filter=w, strides=[1,2, 2,1], padding='SAME',
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
        pool2= tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        gconv_indices, gconv_shape_info, w_shape = gconv2d_util(h_input='C4', h_output='C4', in_channels=8, out_channels=2, ksize=5)
        w = tf.get_variable('w3', shape=w_shape)
        conv3 = gconv2d(input=conv2, filter=w, strides=[1,1,1,1], padding='SAME',
            gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
        conv3 = tf.reshape(conv3, conv3.get_shape().as_list()[:3] + [4] + [out_channels])
        conv3 = tf.reduce_mean(conv3, axis=3)
        pool3= tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
        pool3_flat = tf.layers.flatten(pool3)
        u = pool3_flat
        u =tf.layers.dense(inputs=pool3_flat, units=output_dim, activation=tf.nn.relu, trainable=trainable)
        tf.add_to_collection('conv_output2', conv2)
        return u

@gin.configurable
def eq_cnn_fn(x, output_dim=10, trainable=True, group='C4', num_filters=2):
    nchannels = x.shape[3]
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(h_input='Z2', h_output='C4', in_channels=nchannels, out_channels=2, ksize=5)
    w = tf.get_variable('w1', shape=w_shape)
    
    conv1 = gconv2d(input=x, filter=w,
        strides=[1,1,1,1],
        padding='SAME',
               gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
    tf.add_to_collection('conv_output1', conv1)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    
    # pool1 = layers.Dropout(0.25)(pool1)
    out_channels=2
    gconv_indices, gconv_shape_info, w_shape = gconv2d_util(
            h_input='C4', h_output='C4', in_channels=2, out_channels=out_channels, ksize=5)
    w = tf.get_variable('w2', shape=w_shape)
    conv2 = gconv2d(input=conv1, filter=w, strides=[1,1,1,1], padding='SAME',
        gconv_indices=gconv_indices, gconv_shape_info=gconv_shape_info)
    conv2 = tf.reshape(conv2, conv2.get_shape().as_list()[:3] + [4] + [out_channels])
    conv2 = tf.reduce_mean(conv2, axis=3)
    conv2 = tf.reshape(conv2, conv2.get_shape().as_list()[:3] + [out_channels])
    pool2= tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    pool2_flat = tf.layers.flatten(pool2)
    u = pool2_flat
    print(u.shape)
    u =tf.layers.dense(inputs=pool2_flat, units=output_dim, activation=tf.nn.relu, trainable=trainable)
    tf.add_to_collection('conv_output2', conv2)
    return u 
