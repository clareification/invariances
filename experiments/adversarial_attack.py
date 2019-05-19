import math
import numpy as np
import tensorflow as tf
import os
import itertools
# import variational_model as vm
import pickle as pkl
from keras.utils.np_utils import to_categorical
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import matplotlib.pyplot as plt
# Load adversarial example library
from cleverhans import initializers
from cleverhans.attacks import SaliencyMapMethod, FastGradientMethod
from cleverhans.serial import NoRefModel

from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
from cleverhans.dataset import CIFAR10, MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.train import train
from cleverhans.utils import pair_visual, other_classes, AccuracyReport, set_log_level
from cleverhans.utils_tf import model_eval, model_argmax

class ModelRandomConvolutional(NoRefModel):
	"""
	A simple model that uses only convolution and downsampling---no batch norm or other techniques that can complicate
	adversarial training.
	"""
	def __init__(self, scope, nb_classes, nb_filters, input_shape, perturb=None, perturb_scale=0.1, samples=5, **kwargs):
		del kwargs
		NoRefModel.__init__(self, scope, nb_classes, locals())
		self.samples = samples
		self.nb_filters = nb_filters
		self.input_shape = input_shape
		self.perturb = perturb
		self.perturb_scale = perturb_scale
		# Do a dummy run of fprop to create the variables from the start
		self.fprop(tf.placeholder(tf.float32, [128] + input_shape))
		# Put a reference to the params in self so that the params get pickled
		self.params = self.get_params()
		self.samples = samples
		print("Samples:\n \n  ", self.samples)

	def forward(self, x, **kwargs):
		del kwargs
		conv_args = dict(
			activation=tf.nn.leaky_relu,
			kernel_initializer=initializers.HeReLuNormalInitializer,
			kernel_size=3,
			padding='same')
		if self.perturb:
			y = x + self.perturb_scale*self.perturb([128] + self.input_shape)
			print("PERTURBING w/ scale: \n", self.perturb_scale)
			if self.perturb==tf.random.uniform: y = y - self.perturb_scale/2
		else: y = x
		with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
			log_resolution = int(round(
				  math.log(self.input_shape[0]) / math.log(2)))
			for scale in range(log_resolution - 2):
				y = tf.layers.conv2d(y, self.nb_filters << scale, **conv_args)
				y = tf.layers.conv2d(y, self.nb_filters << (scale + 1), **conv_args)
				y = tf.layers.average_pooling2d(y, 2, 2)
			y = tf.layers.conv2d(y, self.nb_classes, **conv_args)
			logits = tf.reduce_mean(y, [1, 2])
			return logits
	def fprop(self, x, **kwargs):
		logits = tf.reduce_mean(tf.map_fn(lambda y: self.forward(x, **kwargs), 2.0*tf.ones(self.samples)), axis=0)		
		return {self.O_LOGITS: logits,
					  self.O_PROBS: tf.nn.softmax(logits=logits)}


# Train cifar10 model
nb_filters = 64
nb_epochs = 10
batch_size = 128
learning_rate=0.001

def do_eval(preds, x_set, y_set, report_key, is_adv=None):
	acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
	setattr(report, report_key, acc)
	if is_adv is None:
	  report_text = None
	elif is_adv:
	  report_text = 'adversarial'
	else:
	  report_text = 'legitimate'
	if report_text:
	  print('Test accuracy on %s examples: %0.4f' % (report_text, acc))
def evaluate():
      do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)
# Set up training data
config_args = {}
data = MNIST(train_start=0, train_end=50000,
				 test_start=0, test_end=10000)
dataset_size = data.x_train.shape[0]
dataset_train = data.to_tensorflow()[0]
dataset_train = dataset_train.map(
  lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
dataset_train = dataset_train.batch(128)
dataset_train = dataset_train.prefetch(16)
x_train, y_train = data.get_set('train')
x_test, y_test = data.get_set('test')
nb_classes = y_test.shape[1]

train_params = {
	  'nb_epochs': nb_epochs,
	  'batch_size': batch_size,
	  'learning_rate': learning_rate
  }
eval_params = {'batch_size': batch_size}
fgsm_params = {
	  'eps': 0.2,
	  'clip_min': 0.,
	  'clip_max': 1.
  }

jsma_params = {'theta': 1., 'gamma': 0.1,
                 'clip_min': 0., 'clip_max': 1.,
                 'y_target': None}

img_rows, img_cols, nchannels = x_test.shape[1:4]

x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
										nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

def attack(x):
	return fgsm.generate(x, **fgsm_params)

# Train rotation invariant cifar10 model
averages = {1:[], 5:[]}
# Evaluate robustness
for dist, n_samples  in itertools.product([tf.random.normal, None]*3, [1]):
	model = ModelRandomConvolutional('model'+str(np.random.rand()), nb_classes, nb_filters,
									  input_shape=[img_rows, img_cols, nchannels], perturb=dist, perturb_scale=0.2, samples=n_samples)

	preds = model.get_logits(x)
	print(dist)
	
	report = AccuracyReport()
	sess = tf.Session(config=tf.ConfigProto(**config_args))
	fgsm = FastGradientMethod(model, sess=sess)

	def attack(x):
                return fgsm.generate(x, **fgsm_params)
	loss = CrossEntropy(model, attack=attack, smoothing=0.1)
	sess.run(tf.global_variables_initializer())

	rng = np.random.RandomState([2017, 8, 30])
	train(sess, loss, None, None,
		  dataset_train=dataset_train, dataset_size=dataset_size,
		  evaluate=evaluate, args=train_params, rng=rng,
		  var_list=model.get_params())


	accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
	print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

	print('--------------------------------------')
	model.perturb_scale=0.1
	model.samples=5
	model.perturb = None
	print(model.perturb_scale)
	
	adv_x = fgsm.generate(x, **fgsm_params)
	preds_adv = model.get_logits(adv_x)
	# Evaluate the accuracy of the MNIST model on adversarial examples
        	
	acc = do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True)
	preds_rand = model.get_logits(x + 0.1*tf.random.normal([128, 28,28,1]))
	acc_rand = do_eval(preds_rand, x_test, y_test, 'clean_train_rand_eval', True)
	print('Now evaluating with random perturbation')
	fgsm2 = FastGradientMethod(model, sess=sess)
	# model.perturb = dist
	preds_adv = model.get_logits(0.1*tf.random.normal([128, 28,28,1]) + fgsm2.generate(x,**fgsm_params)) 
	new_acc = do_eval(preds_adv, x_test, y_test, 'blah', True)
	averages[n_samples].append(acc)
print(averages)
