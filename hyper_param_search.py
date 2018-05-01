import os
import numpy as np
import tensorflow as tf
import deepchem as dc
benchmark import run_benchmark


np.set_seed(123)
tf.set_random_seed(123)

datasets = ['qm7', 'qm7b', 'qm8', 'qm9']
models = ['rf_regression', 'krr', 'graphconvreg', 'weave_regression']

for dataset in datasets:
    for model in models:
    run_benchmark(
        datasets=[dataset],
        split='random', 
        model=model, 
        featurizer=None, # default
        n_features=0, # default
        out_path=os.path.join('.', 'benchmark'),
        hyper_parameters=None, # default
        hyper_param_search=True,
        max_iter=20, # default
        search_range=4, # default,
        test=False,
        reload=True
    )