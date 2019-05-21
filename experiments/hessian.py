# Compare a discrete set of models.
import itertools
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
NUM_FILTERS=4


def build_hvp(x, t, loss, model, direc=None, dim=1, num_filters=4):
    tf.reset_default_graph()
    x_new = tf.constant(x)
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        y =  model(x_new, output_dim=t.shape[1], num_filters=num_filters)
    direc=direc
    l = loss(t, y)
    vs = tf.trainable_variables()
    print(vs)
    shapes = [w.get_shape().as_list() for w in vs]
    v_reshaped = []
    res = [np.zeros(vv.get_shape().as_list(), dtype=np.float32) for vv in vs]
    v_placeholder = tf.placeholder(dtype=tf.float32, shape=[dim])
    cur=0
    for s in shapes:
        v_reshaped.append(tf.cast(tf.reshape(v_placeholder[cur:np.prod(s) + cur], s), tf.float32))
        cur += np.prod(s)
    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        operation = hvp(l, vs, v_reshaped)
        predict_op = model(x_new, output_dim=10, num_filters=num_filters)


    def apply_hvp(vect):
        sv = tf.train.Supervisor(logdir=direc)
        with sv.managed_session() as sess:
            vector_product, eval_loss, preds = sess.run([operation, l, y], feed_dict={v_placeholder:vect})
            # print(eval_loss, preds[0], t[0], "Evaluated loss")
            for w in range(len(tf.trainable_variables())):
                vi = vector_product[w]
                res[w] = vi
        return np.concatenate([g.reshape(-1, 1) for g in res], axis=0)
    return apply_hvp
        
def approximate_hessian(X, Y, loss_fn, model_fn, model_dir, num_eigs=50, num_filters=4):
    x = tf.constant(X)
    with tf.variable_scope("", reuse=tf.AUTO_REUSE):
        outputs = model_fn(x, output_dim=10, num_filters=num_filters)
    shapes = [w.get_shape().as_list() for w in tf.trainable_variables()]
    cur = 0
    for s in shapes:
        cur += np.prod(s)
    shape = (cur, cur)
    op = build_hvp(X, Y, loss_fn, model_fn, model_dir, dim=cur, num_filters=num_filters)

    A = linalg.LinearOperator(shape, matvec = op)
    eigs = linalg.eigsh(A, k=num_eigs)
    eig_vals = eigs[0]
    print(eig_vals)
    log_det = np.sum(np.log(eig_vals))
    #eig_vals = op(np.ones(cur))
    #print(np.linalg.norm(eig_vals)/cur)
    print(eig_vals)
    return eig_vals

if __name__ == "__main__":
    filename='data/rotatedp4.pickle'
    with open(filename, 'rb') as f:
        d = pkl.load(f)
        unambiguous_X = d['train_ims'][:60000]
        unambiguous_Y = d['train_labels'][:60000]
        es = d['test_ims']
        ls = d['test_labels']

    float_type=np.float64
    use_cifar = False
    class Data:
        input_dim = 784
        Nclasses = 10
        X =  np.reshape(unambiguous_X, (-1, 28,28,1))
        Y = to_categorical(unambiguous_Y, 10)
        Xtest = np.reshape(es, (-1, 28, 28, 1))
        Ytest = to_categorical(ls, 10)
    fn = cnn_fn
    print("Approx")
    evs = []
    ckpt_dir='results/p4_4_filters/noav2filter/'
    
    for i, ckpt_dir in itertools.product(range(5), ['results/p4_4_filters/noav2filter/', 'results/noav_4_filters/noav2filter/', 'results/noav_64_filters/noav2filter/', 'results/p4_64_filters/noav2filter/']):
        ckpt_dir_i =ckpt_dir+str(i)
        tf.reset_default_graph()
        filters = 64 if '64' in ckpt_dir else 4
        print(ckpt_dir, filters)
        evs.append(approximate_hessian(Data.X[:1000], Data.Y[:1000], tf.losses.mean_squared_error, fn, ckpt_dir_i, num_eigs=10, num_filters=filters))
    
    print(evs)
