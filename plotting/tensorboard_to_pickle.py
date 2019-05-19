import tensorflow as tf
import numpy as np
import pickle as pkl
import os



# Returns a dictionary of form {step:value} for given attribute
def summary_dictionary(directory_name, attribute):
	event_file_names = list(filter(lambda x : 'event' in x, os.listdir(directory_name)))
	event_dict = {}
	for file_name in event_file_names:
		t = tf.train.summary_iterator(os.path.join(directory_name, file_name))
		for e in t:
			s = e.step
			for v in e.summary.value:
				if v.tag==attribute:
					event_dict[s] = v.simple_value

	if event_dict == {}:
		return None
	return event_dict



event_dir = "tdlfashion/eqv/1"
def dict_from_dir(event_dir):
	event_file_names = list(filter( lambda x: 'event' in x, os.listdir(event_dir)))
	ds = list(map(lambda x : summary_dictionary(event_dir + "/" + x, 'data_fit'), event_file_names))
	return ds
# av  av4xfilters  eqv  noav  noav4xfilters
for direc in ["eqv", "noav" , "noav2filter"]:
	for i in range(5):
		directory = "fashiontdl/cross_entropy/" + direc + "/" + str(i)
		save_name = "fashiontdl/cross_entropy/" + direc + str(i) + '.pkl'
		curr_dict = summary_dictionary(directory, 'data_fit')
		print(save_name, len(curr_dict.keys()))
		with open(save_name, 'wb') as file:
			pkl.dump(curr_dict, file)
