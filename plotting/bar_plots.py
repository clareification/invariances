import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


av4x = [[0.861375], [0.85725], [0.86655], [0.855875]]
av = [[0.8306], [0.81535], [0.8201], [0.81245], [0.8134] ]
noav = [[0.78045], [0.77375], [0.779975], [0.778025], [0.7698]]
noav4x = [[0.837925], [0.84205], [0.8367], [0.836425], [0.840025]]
eqv = [[0.84535], [0.850175],[0.846225], [0.851275], [0.847225]]

means = []
varis = []
for m in [av4x, av, noav, noav4x, eqv]:
	means.append(np.mean(m))
	varis.append(np.var(np.ones(len(m)) - m))

for method, label, i in zip([ eqv, av4x, noav4x, av, noav], ['G-CNN', 'FA-CNN, 8 filters', 'CNN 8 filters', 'CNN 2 filters', 'FA-CNN, 2 filters' ], range(5)):
	print(np.mean(method), np.var(np.ones(len(method)) - method))
	plt.bar(i, 1 - np.mean(method), yerr=np.var(np.var(np.ones(len(method)) - method)), label=label)
plt.legend()
plt.xticks([])
plt.title("Test error on rotated fashionMNIST")
plt.show()