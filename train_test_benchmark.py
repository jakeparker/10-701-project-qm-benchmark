# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 14:25:40 2017
@author: Zhenqin Wu
"""
from __future__ import division
from __future__ import unicode_literals

import os
import time
import csv
import numpy as np
import tensorflow as tf
import deepchem
from deepchem.molnet.check_availability import CheckFeaturizer, CheckSplit
from deepchem.molnet.preset_hyper_parameters import hps

from qm7_datasets import load_qm7_from_mat
from qm7_datasets import load_qm7b_from_mat
from qm8_datasets import load_qm8
from qm9_datasets import load_qm9

from run_train_test_benchmark_models import train_test_benchmark_regression

def run_train_test_benchmark(datasets,
                         model,
                         split=None,
                         frac_train=0.8,
                         metric=None,
                         featurizer=None,
                         n_features=0,
                         out_path='.',
                         hyper_parameters=None,
                         reload=True,
                         seed=123):
"""
"""
for dataset in datasets:
    if dataset in [
        'qm7', 'qm7b', 'qm8', 'qm9'
    ]:
      mode = 'regression'
      if metric == None:
        metric = [
            deepchem.metrics.Metric(deepchem.metrics.pearson_r2_score, np.mean)
        ]
    else:
      raise ValueError('Dataset not supported')

    if featurizer == None and isinstance(model, str):
      # Assigning featurizer if not user defined
      pair = (dataset, model)
      if pair in CheckFeaturizer:
        featurizer = CheckFeaturizer[pair][0]
        n_features = CheckFeaturizer[pair][1]
      else:
        continue

    if not split in [None] + CheckSplit[dataset]:
      continue

    loading_functions = {
        'qm7': load_qm7_from_mat,
        'qm7b': load_qm7b_from_mat,
        'qm8': load_qm8,
        'qm9': load_qm9,
    }

    print('-------------------------------------')
    print('Train Test Benchmark on dataset: %s' % dataset)
    print('-------------------------------------')
    # loading datasets
    if split is not None:
      print('Splitting function: %s' % split)
      tasks, train_test_dataset, transformers = loading_functions[dataset](
          featurizer=featurizer, split=split, frac_train=frac_train, reload=reload)
    else:
      tasks, train_test_dataset, transformers = loading_functions[dataset](
          featurizer=featurizer, frac_train=frac_train, reload=reload)

    train_dataset, test_dataset = train_test_dataset

    time_start_fitting = time.time()
    train_score = {}
    test_score = {}

    if hyper_parameters is None:
        hyper_parameters = hps[model]
    if isinstance(model, str):
        if mode == 'regression':
        train_score, test_score = train_test_benchmark_regression(
            train_dataset,
            test_dataset,
            tasks,
            transformers,
            n_features,
            metric,
            model,
            hyper_parameters=hyper_parameters,
            seed=seed)

    time_finish_fitting = time.time()

    with open(os.path.join(out_path, dataset + '_' + model + '_' + str(frac_train) + '_results.csv'), 'a') as f:
      writer = csv.writer(f)
      model_name = list(train_score.keys())[0]
      for i in train_score[model_name]:
        output_line = [
            dataset,
            str(split), mode, model_name, i, 'train',
            train_score[model_name][i]
        ]
        output_line.extend(['test', test_score[model_name][i]])
        output_line.extend(
            ['time_for_running', time_finish_fitting - time_start_fitting])
        writer.writerow(output_line)