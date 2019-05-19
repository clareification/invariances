# Compare a discrete set of models.

import numpy as np
import tensorflow as tf
import os
import variational_model as vm
import pickle as pkl
from keras.utils.np_utils import to_categorical
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from cleverhans.dataset import CIFAR10


tf.logging.set_verbosity(tf.logging.ERROR)

filename='rotatedfashionp4.pickle'
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
    
    if use_cifar:
        data = CIFAR10()
        Nclasses = 10
        X, Y = data.get_set('train')
        Xtest, Ytest = data.get_set('test')
        img_rows, img_cols, nchannels = Xtest.shape[1:4] 
        input_dim =img_rows*img_cols*nchannels
        print(img_rows, img_cols, nchannels)	
# dvm = vm.DeepVM(20, 10, averaging=a, ckpt_dir=None, train_features=False)
# features = dvm.feature_map(Data.Xtest)
# lb = dvm.lower_bound(features, Data.Ytest)
test_accuracies={}
for d in ['/tmp/cefmnisteqv1stride']:
	print("Directory: ", d)
	#if 'fashiontdl/av' in d:
	#	a = 'p4'
	#else:
	a = None
	if 'fmnist' in d:
		group="C4"
	else:
		group=None
	tf.reset_default_graph()
	for i in ['0', '1', '2']: #, '3', '4']:
		tf.reset_default_graph()
				
		print("Training iteration:" , i)
		direc= d + '/' + i		
		tf.reset_default_graph()
		dvm = vm.DeepVM(60, 10, averaging=a, ckpt_dir=direc, train_features=False, equivariant=group)
		s = tf.Session()
				
		with tf.variable_scope(dvm.name, reuse=tf.AUTO_REUSE):
			features = dvm.feature_map(Data.Xtest)
		preds = dvm.predict(features)
		targets = tf.cast(Data.Ytest, tf.float64)
		print(tf.trainable_variables())
		x = dvm.model_fit(preds, targets)
		ap = tf.argmax(preds, axis=1)
		at = tf.argmax(targets, axis=1)
		lb = dvm.lower_bound(features, targets)
		kl = dvm.KL()
		fv = dvm.variance(features)
		sv = tf.train.Supervisor(logdir=direc)
		with sv.managed_session() as s:
			# s.run(tf.global_variables_initializer())
			print(s.run([x, ap, at, lb, fv, kl]))
		# print(s.run([dvm.sigma, dvm.KL()]))
			preds = s.run(ap).reshape([-1])
			targets = s.run(at).reshape([-1])
		print(preds.shape, targets.shape)
		acc = np.sum(np.equal(preds, targets).astype(float))/preds.shape
		test_accuracies[direc] = acc
		print(acc)
with open("test_accuracies_cerfp4.pkl", "wb") as dumpfile:
	pkl.dump(test_accuracies, dumpfile)
