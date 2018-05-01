import os
import pickle
import numpy as np
import tensorflow as tf
import deepchem as dc
from train_test_benchmark import run_train_test_benchmark

np.random.seed(123)
tf.set_random_seed(123)

datasets = ['qm7', 'qm7b', 'qm8', 'qm9']
models = ['rf_regression', 'krr', 'graphconvreg', 'weave_regression']
fracs = [float(x+1)/10 for x in range(8)]

for dataset in datasets:
  for model in models:
    # load hyper_parameters from pickle file creatd by `hyper_param_search.py`
    # - molenet params: dataset + model + '.pkl'
    # - hyper param search:  dataset + '_' + model + '_hyper_parameters.pkl'
    
    with open(os.path.join('.', 'pickle', dataset + model + '.pkl'), 'w') as f:
        hyper_parameters = pickle.load(f)
    for frac in fracs:
      run_train_test_benchmark(datasets=[dataset],
                               split='random', 
                               frac_train=frac,
                               model=model, 
                               out_path=os.path.join('.', 'benchmark'),
                               hyper_parameters=hyper_parameters,
                               reload=True)