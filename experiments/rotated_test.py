import numpy as np
import pickle as pkl
import tensorflow as tf
from scipy.cluster.vq import kmeans2
from tensorflow.examples.tutorials.mnist import input_data
import sys
import invariant_models as iv
sys.path.append('~/Documents/Github/spatial-transformer-network')
from stn import spatial_transformer_network as transformer
import gzip
from tensorflow.examples.tutorials.mnist import input_data
import rotation_augmentation as ra
augmented_test = False
float_type=np.float64

def linear_classifier(features):
  return tf.layers.dense(inputs = features, units=10, activation=None)


def evaluate_cnn_model(input_data, input_labels, ckpt_dir, feature_av=False):
    print(ckpt_dir)
    rot = lambda x : iv.rotate(x, -np.pi, np.pi)
    feature = lambda x : tf.cast(iv.cnn_fn(tf.cast(rot(x), tf.float32), 20), float_type) 
    f = lambda x : tf.cast(iv.cnn_fn(tf.cast(x, tf.float32), 20), float_type) 
    orbit = lambda x : tf.reduce_sum(tf.map_fn(lambda y : feature(x), tf.ones(100, dtype=tf.float64)),axis=0)/100.0
    with tf.variable_scope('forward', reuse=tf.AUTO_REUSE): 
        if feature_av:
            features = orbit(input_data)
        else:
            features = f(input_data)
    predictions = linear_classifier(features)
    sv = tf.train.Supervisor(logdir=ckpt_dir)
    print("LOGDIR:", sv.save_path)
    with sv.managed_session() as sess:
        predictions = sess.run(predictions)
    preds = np.argmax(predictions, axis=1)
    print("Predictions: ", predictions)
    ts = input_labels
    ts = np.reshape(ts, [-1])
    preds = np.reshape(preds, [-1]) 
    print(preds[:10], ts[:10])
    print(np.sum(np.equal(preds, ts).astype(float))/ts.size)
    print(ts.size)
def check_features(input_data, ckpt_dir, feature_av=False):
    f = lambda x : tf.cast(iv.cnn_fn(tf.cast(x, tf.float32), 20), float_type)
   
    rot = lambda x : iv.rotate(x, -np.pi, np.pi)
    feature = lambda x : tf.cast(iv.cnn_fn(tf.cast(rot(x), tf.float32), 20), float_type) 
    orbit = lambda x : tf.reduce_sum(tf.map_fn(lambda y : feature(x), tf.ones(100, dtype=tf.float64)),axis=0)/100.0 
    with tf.variable_scope('forward', reuse=tf.AUTO_REUSE):
        if feature_av:
            features = orbit(input_data)
        else: 
            features = f(input_data)
    sv = tf.train.Supervisor(logdir=ckpt_dir)
    print("LOGDIR:", sv.save_path)
    with sv.managed_session() as sess:
        features = sess.run(features)
    print(features)

if __name__ == "__main__":
    mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=False)

    
    class Mnist:
        input_dim = 784
        Nclasses = 10
        X = mnist.train.images.astype(float)
        Y = mnist.train.labels.astype(float)[:, None]
        Xtest = mnist.test.images.astype(float)
        Ytest = mnist.test.labels.astype(float)[:, None]
        # Xtest, Ytest = ra.unambiguous_mnist(Xtest, Ytest)
        X, Y = ra.unambiguous_mnist(X, Y)
        Xtest, Ytest = ra.unambiguous_mnist(Xtest, Ytest)
    # mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=False)
    # es, ls = expand_training_data(Mnist.Xtest, Mnist.Ytest)
    with open('rotatedp4.pickle', 'rb') as f:
        d = pkl.load(f)
    es = d['test_ims'][:10000]   
    ls = d['test_labels'][:10000] 
    with tf.Graph().as_default():
        print("Evaluating first model") 
        evaluate_cnn_model(es, ls, '/home/clare/trained_models/learn/faiew5gwjiji', feature_av=True)
    with tf.Graph().as_default():
        evaluate_cnn_model(Mnist.Xtest, Mnist.Ytest, '/home/clare/trained_models/learn/faiew5gwjiji', feature_av=True)

    with tf.Graph().as_default():
        evaluate_cnn_model(Mnist.Xtest, Mnist.Ytest, '/home/clare/trained_models/learn/faiew5gwjiji', feature_av=False)


    with tf.Graph().as_default():
        evaluate_cnn_model(es,ls,  '/home/clare/trained_models/learn/faiew5gwjiji', feature_av=False)
