import os
import pickle
import numpy as np
import tensorflow as tf
import deepchem as dc
train_test_benchmark import run_train_test_benchmark

np.set_seed(123)
tf.set_random_seed(123)

datasets = ['qm7', 'qm7b', 'qm8', 'qm9']
models = ['rf_regression', 'krr', 'graphconvreg', 'weave_regression']
fracs = range(0.1, 0.8, 0.1)

for dataset in datasets:
    for model in models:
    # load hyper_parameters from pickle file creatd by `hyper_param_search.py`
    hyper_parameters = pickle.load(dataset + '_' + model + '_hyper_parameters.pkl')
    for frac in fracs:
        run_train_test_benchmark(
            datasets=[dataset],
            split='random', 
            frac_train=frac,
            model=model, 
            out_path=os.path.join('.', 'benchmark'),
            hyper_parameters=hyper_parameters,
            reload=True
        )