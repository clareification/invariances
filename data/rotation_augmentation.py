import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pickle as pkl
from tensorflow.examples.tutorials.mnist import input_data
import sys
import gzip
from tensorflow.examples.tutorials.mnist import input_data
from scipy import ndimage
from scipy.misc import imsave

sys.path.append('~/Documents/Github/spatial-transformer-network')
from stn import spatial_transformer_network as transformer


def extract_data(filename, num):
	print('Extracting', filename)
	#unzip data
	with gzip.open(filename) as bytestream:
		bytestream.read(16)
		buf = bytestream.read(28 * 28 * num)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = data.reshape(num, 28, 28,1) #reshape into tensor
	return data

def extract_labels(filename, num):
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(num)
		labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
	return labels

# Labels is list of integers 0-9
def unambiguous_mnist(images, labels):
	# 2, 5, 6, 9 are all rotationally ambiguous, so eliminate them.
	ambigs = [2,5,6,9]
	unambiguous_images = []
	unambiguous_labels = []
	for i, l in zip(images, labels):
		if l[0] not in ambigs:
			unambiguous_images.append(i)
			unambiguous_labels.append(l)
	return np.array(unambiguous_images), np.array(unambiguous_labels)



def expand_training_data(images, labels, discrete=False):
	""" Inputs:
	images: original array of image data
	labels: labels (integer in [0, 9]) corresponding to images
	discrete: boolean, discrete orbit vs sampled
	Outputs: Augmented dataset
	"""

	np.random.seed(0)
	num_copies = 3
	expanded_images = np.zeros([labels.shape[0]*(num_copies+1), 784])
	expanded_labels = []
	k = 0 # counter
	for x, y in zip(images, labels):
		#print(x.shape)
		k = k+1
		if k%100==0:
			print ('expanding data : %03d / %03d' % (k,np.size(images,0)))

		# register original data
		expanded_images[(1 + num_copies)*(k-1)] = np.reshape(x, (784,))

		bg_value = 0.0 # this is regarded as background's value black
		#print(x)
		image = np.reshape(x, [-1, 28])

		for i in range(num_copies+ 1):
			# rotate the image with random degree
			if not discrete:
				angle = np.random.randint(-90,90,1)
			else:
				angle = 90*(i % 4)
			new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

			# register new training data
			expanded_images[(1 + num_copies)*(k-1) + i] = np.reshape(new_img, (784,))
			expanded_labels.append(y)

	# return them as arrays

	expandedX=expanded_images
	expandedX = np.reshape(expandedX, [-1, 784])
	expandedY=np.asarray(expanded_labels)
	return expandedX.astype(np.float32), expandedY


#----------------------- Main ------------------------------------------

if __name__ == "__main__":
	print("test")
	# Save rotated and unambiguous MNIST train and test set
	#  Step 1: disambiguate
	mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=False)
	class Mnist:
		input_dim = 784
		Nclasses = 10
		X = mnist.train.images.astype(float)
		Y = mnist.train.labels.astype(float)[:, None]
		Xtest = mnist.test.images.astype(float)
		Ytest = mnist.test.labels.astype(float)[:, None]
	class FashionMnist:
		input_dim = 784
		Nclasses = 10
		(X, Y), (Xtest, Ytest) = keras.datasets.fashion_mnist.load_data()
	test_ims, test_labels = FashionMnist.Xtest, FashionMnist.Ytest
	train_ims, train_labels = FashionMnist.X, FashionMnist.Y
	print(Mnist.X[0])
	print(test_ims.shape, test_labels.shape)
	# Expand to include rotations

	rot_test_ims, rot_test_labels = expand_training_data(test_ims, test_labels, discrete=True)
	rot_train_ims, rot_train_labels = expand_training_data(train_ims, train_labels, discrete=True)
	rotated_dict = {'test_ims': rot_test_ims, 'test_labels': rot_test_labels,
	'train_ims': rot_train_ims, 'train_labels': rot_train_labels}
	with open('rotatedfashionp4.pickle', 'wb') as f:
		pkl.dump(rotated_dict, f, pkl.HIGHEST_PROTOCOL)


