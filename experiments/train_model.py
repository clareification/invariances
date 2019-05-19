import numpy as np
import tensorflow as tf
import os
import gin.tf
import matplotlib.pyplot as plt
import pickle as pkl
from keras.utils.np_utils import to_categorical
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

from cleverhans.dataset import CIFAR10
import itertools
import sys
sys.path.append('../')
from model_comparison.models import models
from model_comparison.models import transformations
from model_comparison.models import networks

tf.logging.set_verbosity(tf.logging.ERROR)
def flip_logits(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)


@gin.configurable
class Runner():
    def __init__(ckpt_dir, equivariant, loss_fn, dataset_file='rotatedfashionp4.pickle', net=networks.cnn_fn, num_iters=1):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.equivariant = equivariant
        self.loss_fn = loss_fn
        self.dataset_file = dataset_file
        self.num_iters = num_iters

        # Initialize dataset

        with open(filename, 'rb') as f:
            d = pkl.load(f)
            d['']


    def run_one_iteration(self, i):
        iteration_dir = os.path.join(self.ckpt_dir, i)




ckpt_dir = 'fashiontdl/test/'
eqv = True
latest_ckp = tf.train.latest_checkpoint(ckpt_dir)
loss_fn = flip_logits
# print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
num_iters = 5


filename='data/rotatedp4.pickle'
with open(filename, 'rb') as f:
    d = pkl.load(f)
    unambiguous_X = np.reshape(d['train_ims'][:60000], [-1, 28,28,1])
    unambiguous_Y = d['train_labels'][:60000]
    es = np.reshape(d['test_ims'][:1000], [-1, 28,28,1])
    ls = d['test_labels'][:1000]

float_type=np.float64
# (unambiguous_X, unambiguous_Y), (es, ls) = tf.keras.datasets.mnist.load_data()
use_cifar=False
class Data:
    input_dim = 784
    Nclasses = 10
    X =  np.reshape(unambiguous_X, (-1, 28, 28, 1))
    Y = to_categorical(unambiguous_Y, 10)
    Xtest = np.reshape(es, (-1, 28, 28, 1))
    Ytest = to_categorical(ls, 10)
    if use_cifar:
        data = CIFAR10()
        Nclasses = 10
        X, Y = data.get_set('train')
        Xtest, Ytest = data.get_set('test')
        img_rows, img_cols, nchannels = Xtest.shape[1:4]
        input_dim =img_rows*img_cols*nchannels
        print(img_rows, img_cols, nchannels)

for i, eqv in itertools.product(range(num_iters), [None]):
    eqv_dir = eqv if eqv else 'noav2filter'
    ckpt_dir_i = ckpt_dir + '/' + eqv_dir + '/' + str(i)
    dvm =models.FAModel(networks.cnn_fn, Data.X, Data.Y,
        Data.Xtest, Data.Ytest, averaging=None, ckpt_dir=ckpt_dir_i, batch_size=32)

    # dvm = models.DeepVM(10, 10, averaging=None, ckpt_dir=ckpt_dir, train_features=True)
    # with tf.variable_scope(dvm.name, reuse=tf.AUTO_REUSE):
    #     features = dvm.feature_map(tf.cast(Data.Xtest, tf.float32))

    s = tf.Session()
    #features = dvm.feature_map(Data.X[:1])
    # x = dvm.lower_bound(features, tf.cast(Data.Y[:100], tf.float64))
    s.run(tf.global_variables_initializer())

    
    print(tf.trainable_variables())
    dvm.optimize(steps=10)
    # dvm.optimizeDataFit(Data.X, Data.Y, steps=5000)
    ps = dvm.predict(Data.X[0])
    print("Done optimizing")
    print("SHAPE:", ps.shape)
    tf.reset_default_graph()
# dvm =models.DeepVM(60, 10, averaging=None, ckpt_dir=ckpt_dir)

# with tf.variable_scope(dvm.name, reuse=tf.AUTO_REUSE):
#     features = dvm.feature_map(Data.Xtest)
# preds = dvm.predict(features)
# targets = tf.cast(Data.Ytest, tf.float64)
# print(tf.trainable_variables())
# x = dvm.model_fit(preds, targets)
# ap = tf.argmax(preds, axis=1)
# at = tf.argmax(targets, axis=1)
# lb = dvm.lower_bound(features, targets)
# kl = dvm.KL()
# fv = dvm.variance(features)
# sv = tf.train.Supervisor(logdir=ckpt_dir)
# with sv.managed_session() as s:
#     # s.run(tf.global_variables_initializer())
#     print(s.run([x, ap, at, lb, fv, kl]))
# # print(s.run([dvm.sigma, dvm.KL()]))
#     preds = s.run(ap).reshape([-1])
#     targets = s.run(at).reshape([-1])
# print(preds.shape, targets.shape)
# print(np.sum(np.equal(preds, targets).astype(float)))
