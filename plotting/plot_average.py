import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

# load data for each class of FA

def smooth(data, window=2):
	smoothed = data.copy()
	for i in range(window, len(data) - window):
		d = data[i-window : i + window]
		smoothed[i] = np.average(data[i-window : i + window])
	return smoothed


average = {}
types = ["eqv", "noav", "noav2filter"]
labels = ["G-CNN", "4xCNN", "CNN"]
filenames = [["cross_entropy/" + t + str(i) + ".pkl" for i in range(5)] for t in types]

dicts = []
colors =['blue', 'purple', 'green', 'red', 'orange']


# 3 lines: mean, 
for files, col, label in zip(filenames, colors, labels):
	means = {}
	maxes = {}
	mins = {}
	list_dict = {}
	for f in files:
		print(f)
		with open(f, 'rb') as file:
			d = pkl.load(file)
			# average.update(d)
			for k in d.keys():
				if k in means.keys():
					print(k)
					v = d[k]
					means[k] = [v] + means[k]
					maxes[k] = max(v, maxes[k])
					mins[k] = min(v, mins[k])
				else:
					means[k] = [d[k]]
					maxes[k] = d[k]
					mins[k] = d[k]
	for k in means.keys():
		if len(means[k]) > 1:
			print(k)
		means[k] = np.average(means[k])
	xs = sorted(means.keys())
	ys = [means[k] for k in xs]
	ys = smooth(ys, window=5)
	lows = [mins[x] for x in xs]
	#lows = smooth(lows, window=5)
	highs = [maxes[x] for x in xs]
	#highs = smooth(highs, window=5)
	plt.plot(xs[5:-5], ys[5:-5], alpha=0.2, color=col)
	plt.plot(xs[5:-5], highs[5:-5], alpha=0.2, color=col)
	# ys = smooth(ys, int(len(ys)/20))
	plt.plot(xs[5:-5], ys[5:-5], color=col, label=label)
plt.xlabel("Training steps")
plt.ylabel("Error on training batch")
plt.legend()
plt.show()

# sort
