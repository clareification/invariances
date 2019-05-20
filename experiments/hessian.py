# Compare a discrete set of models.

import numpy as np
import tensorflow as tf
import os

import pickle as pkl
from keras.utils.np_utils import to_categorical
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import sys
sys.path.append('../')
from tensorflow.python.ops.gradients_impl import _hessian_vector_product as hvp
from tensorflow.python.ops.gradients_impl import gradients_v2
from model_comparison.models.networks import cnn_fn
from scipy.sparse import linalg

tf.logging.set_verbosity(tf.logging.ERROR)

x = tf.random_uniform([32,28,28,1])
y = cnn_fn(x, output_dim=1)
t = tf.random_uniform([32,1])

l = tf.losses.mean_squared_error(t, y)
vs = tf.trainable_variables()

shapes = [w.get_shape().as_list() for w in vs]
dim = np.sum([np.prod(s) for s in shapes])
shape = (dim, dim)

v_vect = np.ones((dim),dtype=np.float32)
v_reshaped=[]
cur=0
for s in shapes:
        v_reshaped.append(tf.reshape(v_vect[cur:np.prod(s) + cur], s))
        cur += np.prod(s)
print(len(v_reshaped), len(vs), v_reshaped[0].shape, vs[0].shape)

s = tf.Session()
s.run(tf.global_variables_initializer())
def apply_hvp(vect):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    shapes = [w.get_shape().as_list() for w in tf.trainable_variables()]
    v_reshaped = []
    res = [np.zeros(vv.get_shape().as_list(), dtype=np.float32) for vv in tf.trainable_variables()]
    cur=0
    for s in shapes:
        v_reshaped.append(tf.cast(tf.reshape(vect[cur:np.prod(s) + cur], s), tf.float32))
        cur += np.prod(s)
    vector_product = sess.run(hvp(l, vs, v_reshaped))
    for w in range(len(tf.trainable_variables())):
        vi = vector_product[w]
        res[w] = vi


    return np.concatenate([g.reshape(-1, 1) for g in res], axis=0)
    


A = linalg.LinearOperator(shape, matvec=apply_hvp)

gs = gradients_v2(l, tf.trainable_variables())


v = np.ones([dim])
v = np.ones((dim), dtype=np.float32)
v2 = np.ones((dim), dtype=np.float32)

h = hvp(l, tf.trainable_variables(), v_reshaped)
h2 = h

grads, hess = s.run([gs, h])
eigs = linalg.eigsh(A, k=5)
print(eigs)
print(len(eigs))
print(len(grads), [grads[i].shape for i in range(len(grads))])
print(len(hess), hess[0].shape)
