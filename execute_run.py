#!/usr/bin/env python3.6
#
# Train and test Graph-CNN models on given combinations of training and test data.
#
# Adapted from https://github.com/mdeff/cnn_graph + https://nbviewer.jupyter.org/github/mdeff/cnn_graph/blob/outputs/usage.ipynb
#

from lib import models, graph, coarsening, utils
import pandas
import numpy as np
import matplotlib.pyplot as plt
import pickle
%matplotlib inline


## load training data (could be a CV-fold):
dt_train = pandas.read_csv(sys.argv[1])
X_train_tmp = dt_train.as_matrix()[:,0:-1]
y_train_tmp = dt_train.as_matrix()[:,-1]

## load test data (could be a CV-fold):
dt_test = pandas.read_csv(sys.argv[2])
X_test = dt_test.as_matrix()[:,0:-1]
y_test = dt_test.as_matrix()[:,-1]

outdir = sys.argv[3]



## constants:
c = y.max() + 1 	# number of classes
assert c == np.unique(y).size
n = len(y) 			# number of samples
d = X.shape[1] 		# number of features
n_val = n // 10 	# size of training validation dataset

## parameters for ML:
params = dict()
params['dir_name']       = 'demo'
params['num_epochs']     = 40 #40
params['batch_size']     = 100
params['eval_frequency'] = 200
# Building blocks.
params['filter']         = 'chebyshev5'
params['brelu']          = 'b1relu'
params['pool']           = 'apool1'
# Architecture.
params['F']              = [32, 64]  # Number of graph convolutional filters.
params['K']              = [20, 20]  # Polynomial orders.
params['p']              = [4, 2]    # Pooling sizes.
#params['M']              = [512, c]  # Output dimensionality of fully connected layers.
params['M']              = [250, c]  # Output dimensionality of fully connected layers.
# Optimization.
params['regularization'] = 5e-4
params['dropout']        = 0.95 #1
params['learning_rate']  = 1e-3
params['decay_rate']     = 0.95
params['momentum']       = 0.9
params['decay_steps']    = n_train / params['batch_size']

# keep a part of the training dataset for validation:
X_val = X[:n_val]
X_train   = X[n_val:]
y_val = y[:n_val]
y_train   = y[n_val:]

# determine graph structure (adjacency from Euclidean distance):
dist, idx = graph.distance_scipy_spatial(X_train.T, k=10, metric='euclidean')
A = graph.adjacency(dist, idx).astype(np.float32)

assert A.shape == (d, d)
#print('d = |V| = {}, k|V| < |E| = {}'.format(d, A.nnz))
plt.spy(A, markersize=2, color='black');
plt.savefig('{}/spy.pdf'.format(outdir))

# prep graphs (and data) for CNN:
graphs, perm = coarsening.coarsen(A, levels=3, self_connections=False)
X_train = coarsening.perm_data(X_train, perm)
X_val = coarsening.perm_data(X_val, perm)
X_test = coarsening.perm_data(X_test, perm)
L = [graph.laplacian(A, normalized=True) for A in graphs]
graph.plot_spectrum(L)
plt.savefig('{}/spectrum.pdf'.format(outdir))


# define CNN:
model = models.cgcnn(L, **params)

# train CNN:
accuracy, loss, t_step = model.fit(X_train, y_train, X_val, y_val)


fig, ax1 = plt.subplots(figsize=(15, 5))
ax1.plot(accuracy, 'b.-')
ax1.set_ylabel('validation accuracy', color='b')
ax2 = ax1.twinx()
ax2.plot(loss, 'g.-')
ax2.set_ylabel('training loss', color='g')
plt.show()
plt.savefig('{}/performance.pdf'.format(outdir))

print('Time per step: {:.2f} ms'.format(t_step*1000))

pred = model.predict(X_test)
res = model.evaluate(X_test, y_test)

print(res[0])




# save results:
with open('{}/predictions.csv'.format(outdir), 'wb') as fp:
	outfile.write(",".join(pred))
with open('{}/performance.csv'.format(outdir), 'wb') as fp:
	outfile.write(",".join(res))
	
# store models:
with open('{}/model.pkl'.format(outdir), 'wb') as fp:
    pickle.dump(model, fp)
with open('{}/perm.pkl'.format(outdir), 'wb') as fp:
    pickle.dump(perm, fp)	
	
	
	