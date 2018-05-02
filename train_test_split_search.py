import os
import pickle
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.molnet.preset_hyper_parameters import hps
from train_test_benchmark import run_train_test_benchmark

np.random.seed(123)
tf.set_random_seed(123)

datasets = ['qm7', 'qm7b', 'qm9']
tasks = dict({'qm7': ['u0_atom'], 'qm7b': [3, 4], 'qm9': ['homo', 'lumo']})
featurizers = ['ECFP', 'CoulombMatrix', 'GraphConv', 'Weave'] 
models = ['rf_regression', 'krr', 'graphconvreg', 'weave_regression', 'dtnn', 'mpnn']
fracs = [float(x+1)/10 for x in range(8)]

for dataset in datasets:
  for model in models:
    try:    
      with open(os.path.join('.', 'pickle', dataset + model + '.pkl'), 'rb') as f:
        hyper_parameters = pickle.load(f)
    except:
      hyper_parameters = hps[model]
    for featurizer in featurizers:
      for task in tasks[model]:
        for frac in fracs:
          run_train_test_benchmark([dataset],
                                   model,
                                   task,
                                   split='random', 
                                   frac_train=frac,
                                   metric=[deepchem.metrics.Metric(deepchem.metrics.mae_score, task_averager=np.mean)],
                                   featurizer=featurizer,
                                   out_path=os.path.join('.', 'benchmark'),
                                   hyper_parameters=hyper_parameters,
                                   reload=False,
                                   seed=123)
