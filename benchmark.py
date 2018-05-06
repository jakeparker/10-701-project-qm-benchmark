import time
import pathlib
import csv
import pickle

import numpy as np
import tensorflow as tf
import deepchem


SEED = 123
np.random.seed(SEED)
tf.set_random_seed(SEED)


def run_benchmark(data, model, tasks, metrics, transformers, n_features, 
                  direction, hyper_parameters, hyper_parameter_search, max_iter, search_range,
                  valid, test, out_path, reload, seed):
  pass

  time_start_fitting = time.time()

  # ...

  time_finish_fitting = time.time()
  scores = dict({
    'train': train_score, 
    'valid': valid_score,
    'test': test_score
  })
  runtime = time_finish_fitting - time.start_fitting
  return scores, hyper_parameters, runtime


def load_data(dataset):
  pass
  return data

def featurize_data(data, featurizer):
  pass
  return featurized_data, transformers, n_features

def benchmark(datasets, featurizers, modes, methods, models, features, tasks, splits, fracs, metrics,
              hyper_parameters_init=None, hyper_parameter_search=True, max_iter=20, search_range=4,
              valid=True, test=True, out_path=pathlib.Path('.'), load_hyper_parameters=False, save_hyper_parameters=False, save_results=True, reload=True, seed=None):
  """
  dataset,feature,mode,method,model,task,split,frac_train,frac_valid,frac_test,metric,train_score,valid_score,test_score,runtime
  """
  if hyper_parameters_init != None:
    assert(load_hyper_parameters == False)
    if all([isinstance(key, str) for key in hyper_parameters_init.keys()]):
      hyper_parameters_lookup = lambda dataset, model: hyper_parameters_init['model']
    elif all([isinstance(key, frozenset) for key in hyper_parameters_init.keys()]):
      hyper_parameters_lookup = lambda dataset, model: hyper_parameters_init[frozenset(['dataset', 'model'])]
    else:
      raise ValueError
  elif load_hyper_parameters:
    hyper_parameters_lookup = lambda dataset, model: with open(pathlib.Path('.') / 'pickle' / dataset + model + '.pkl', 'rb') as f: pickle.load(f)
  else:
    hyper_parameters_lookup = lambda dataset, model: deepchem.molnet.preset_hyper_parameters.hps[model]

  lookup_split_func = dict({
    'index': deepchem.splits.IndexSplitter(),
    'random': deepchem.splits.RandomSplitter(),
    'stratified': deepchem.splits.SingletaskStratifiedSplitter(task_number=0)
  })

  lookup_metric_func = dict({
    'MAE': deepchem.metrics.mae_score,
    'RMSE': deepchem.metrics.rms_score,
    'R2': deepchem.metrics.pearson_r2_score
  })

  lookup_direction = dict({
    'MAE': False, # minimize
    'RMSE': False, # minimize
    'R2': True # maximize
  })

  for dataset in datasets:
    print('-------------------------------------')
    print('Benchmark on dataset: %s' % dataset)
    print('-------------------------------------')
    metric_funcs = map(deepchem.metrics.Metric, [lookup_metric_func[metric] for metric in metrics[dataset]])
    direction = lookup_direction[metrics[dataset][0]]
    data = load_data(dataset) # load dataset
    for featurizer in featurizers:
      print("About to featurize %s dataset using: %s" % (dataset, featurizer))
      featurized_data, transformers, n_features = featurize_data(data, featurizer) # featurize dataset
      for split in splits:
        for frac in fracs:
          split_func = lookup_split_func[split]
          if valid and test:
            frac_train = frac
            frac_valid = floor((1-frac) / 2.0)
            frac_test = ceil((1-frac) / 2.0)
            print('About to split %s dataset into {%d train / %d valid / %d test} sets using %s split' % (dataset, frac_train, frac_valid, frac_test, split))
            train_set, valid_set, test_set = split_func.train_valid_test_split(dataset, frac_train=frac_train, frac_test=frac_test, frac_valid=frac_valid, seed=seed)
          elif valid:
            frac_train = frac
            frac_valid = 1-frac
            frac_test = None
            print('About to split %s dataset into {%d train / %d valid} sets using %s split' % (dataset, frac_train, frac_valid, split))
            test_set = None
            train_set, valid_set = split_func.train_test_split(dataset, frac_train=frac_train, seed=seed)
          elif test:
            frac_train = frac
            frac_valid = None
            frac_test = 1-frac
            print('About to split %s dataset into {%d train / %d test} sets using %s split' % (dataset, frac_train, frac_test, split))
            valid_set = None
            train_set, test_set = split_func.train_test_split(dataset, frac_train=frac_train, seed=seed)
          else:
            frac_train = frac
            frac_valid = None
            frac_train = None
            print('About to split %s dataset into {%d train} set using %s split' % (dataset, frac_train, split))
            valid_set, test_set = None
            train_set, _ = split_func.train_test_split(dataset, frac_train=frac_train, seed=seed)
          split_featurized_data = dict({'train': train_set, 'valid': valid_set, 'test': test_set})
          if len(modes[dataset] == 0): 
            pass
          else:
            for mode in modes[dataset]:
              for method in methods:
                key = frozenset([featurizer, mode, method])
                if models[key] == None:
                  pass
                else:
                  for model in models[key]:
                    n_features = features[frozenset([dataset, featurizer, model])]
                    if tasks[dataset] == None:
                      pass
                    else:
                      hyper_parameters = hyper_parameters_lookup(dataset, model)
                      for task in tasks[dataset]:
                        scores, hyper_parameters, runtime = run_benchmark(split_featurized_data,
                                                                          model,
                                                                          [task],
                                                                          metric_funcs,
                                                                          transformers,
                                                                          n_features, 
                                                                          direction,
                                                                          hyper_parameters,
                                                                          hyper_parameter_search,
                                                                          max_iter,
                                                                          search_range,
                                                                          valid,
                                                                          test,
                                                                          out_path,
                                                                          reload,
                                                                          seed)
                        if save_hyper_parameters:
                          with open(pathlib.Path('.') / 'pickle' / dataset + model + '.pkl', 'wb') as f:
                            pickle.dump(f, hyper_parameters)
                        if save_results:
                          with open(out_path / 'results.csv', 'a') as f:
                            writer = csv.writer(f)
                            output_line = [
                              dataset,
                              feature,
                              mode,
                              method,
                              model,
                              str(task),
                              str(split),
                              str(frac_train),
                              str(frac_valid),
                              str(frac_test),
                              metric,
                              scores['train']
                            ]
                            if valid:
                              output_line.extend([scores['valid']])
                            if test:
                              output_line.extend([scores['test']])
                            output_line.extend([runtime])
                            writer.writerow(output_line)
  return None


# esol solubility measurements
#   - RMSE
#   - ECFP (RF, KRR)
#   - GraphConv (Graph Convolution)
#   - Weave (Weave Regresion)

# feesolv solvation energy mesasurements
#   - RMSE
#   - ECFP (RF, KRR)
#   - GraphConv (Graph Convolution)
#   - Weave (Weave Regresion)

# qm 7b/9 Homo Lumo task (ab-initio calculated energies)
#   - MAE
#   - ECFP (RF, KRR)
#   - CoulombMatrix (DTNN, MPNN, ANI)
#   - GraphConv (Graph Convolution)
#   - Weave (Weave Regresion)

def main():
  datasets = ['qm7b', 'qm9']
  featurizers = ['Raw', 'ECFP', 'CoulombMatrix', 'GraphConv', 'Weave', 'MP', 'BPSymmetryFunction']
  modes = dict({
    'qm7': ['regression', 'coordinates'],
    'qm7b': ['regression', 'coordinates'],
    'qm8': ['regression', 'coordinates'],
    'qm9': ['regression', 'coordinates'],
    'esol': ['regression'],
    'freesolv': ['regression']
  })
  methods = ['conventional', 'graph']
  models = dict({ # { (featurizer, mode, method) --> models }
    frozenset(['Raw', 'regression', 'conventional']): None,
    frozenset(['Raw', 'regression', 'graph']): ['textcnn_regression'],
    frozenset(['Raw', 'coordinates', 'conventional']): None,
    frozenset(['Raw', 'coordinates', 'graph']): None,
    frozenset(['Raw', 'classification', 'conventional']): None,
    frozenset(['Raw', 'classification', 'graph']): None,
    frozenset(['ECFP', 'regression', 'conventional']): ['rf_regression', 'krr'],
    frozenset(['ECFP', 'regression', 'graph']): ['tf_regression'],
    frozenset(['ECFP', 'coordinates', 'conventional']): None,
    frozenset(['ECFP', 'coordinates', 'graph']): None,
    frozenset(['ECFP', 'classification', 'conventional']): None,
    frozenset(['ECFP', 'classification', 'graph']): None,
    frozenset(['CoulombMatrix', 'regression', 'conventional']): ['krr_ft'],
    frozenset(['CoulombMatrix', 'regression', 'graph']): ['tf_regression_ft', 'dtnn'],
    frozenset(['CoulombMatrix', 'coordinates', 'conventional']): None,
    frozenset(['CoulombMatrix', 'coordinates', 'graph']): None,
    frozenset(['CoulombMatrix', 'classification', 'conventional']): None,
    frozenset(['CoulombMatrix', 'classification', 'graph']): None,
    frozenset(['GraphConv', 'regression', 'conventional']): None,
    frozenset(['GraphConv', 'regression', 'graph']): ['graphconvreg'],
    frozenset(['GraphConv', 'coordinates', 'conventional']): None,
    frozenset(['GraphConv', 'coordinates', 'graph']): None,
    frozenset(['GraphConv', 'classification', 'conventional']): None,
    frozenset(['GraphConv', 'classification', 'graph']): None,
    frozenset(['Weave', 'regression', 'conventional']): None,
    frozenset(['Weave', 'regression', 'graph']): ['weave_regression'],
    frozenset(['Weave', 'coordinates', 'conventional']): None,
    frozenset(['Weave', 'coordinates', 'graph']): None,
    frozenset(['Weave', 'classification', 'conventional']): None,
    frozenset(['Weave', 'classification', 'graph']): None,
    frozenset(['MP', 'regression', 'conventional']): None,
    frozenset(['MP', 'regression', 'graph']): ['mpnn'],
    frozenset(['MP', 'coordinates', 'conventional']): None,
    frozenset(['MP', 'coordinates', 'graph']): None,
    frozenset(['MP', 'classification', 'conventional']): None,
    frozenset(['MP', 'classification', 'graph']): None,
    frozenset(['BPSymmetryFunction', 'regression', 'conventional']): None,
    frozenset(['BPSymmetryFunction', 'regression', 'graph']): ['ani'],
    frozenset(['BPSymmetryFunction', 'coordinates', 'conventional']): None,
    frozenset(['BPSymmetryFunction', 'coordinates', 'graph']): None,
    frozenset(['BPSymmetryFunction', 'classification', 'conventional']): None,
    frozenset(['BPSymmetryFunction', 'classification', 'graph']): None
  })
  features = dict({
    frozenset(['qm7', 'ECFP', 'krr']): 1024,
    frozenset(['qm7', 'ECFP', 'rf_regression']): 1024,
    frozenset(['qm7', 'ECFP', 'tf_regression']): 1024,
    frozenset(['qm7', 'CoulombMatrix', 'krr_ft']): 1024,
    frozenset(['qm7', 'CoulombMatrix', 'tf_regression_ft']): [23, 23],
    frozenset(['qm7', 'CoulombMatrix', 'dtnn']): [23, 23],
    frozenset(['qm7', 'GraphConv', 'graphconvreg']): 75,
    frozenset(['qm7', 'Weave', 'weave_regression']): 75,
    frozenset(['qm7', 'BPSymmetryFunction', 'ani']): [23, 4],
    frozenset(['qm7', 'Raw', 'textcnn_regression']): None,

    frozenset(['qm7b', 'CoulombMatrix', 'krr_ft']): 1024,
    frozenset(['qm7b', 'CoulombMatrix', 'tf_regression_ft']): [23, 23],
    frozenset(['qm7b', 'CoulombMatrix', 'dtnn']): [23, 23],

    frozenset(['qm8', 'ECFP', 'krr']): 1024,
    frozenset(['qm8', 'ECFP', 'rf_regression']): 1024,
    frozenset(['qm8', 'ECFP', 'tf_regression']): 1024,
    frozenset(['qm8', 'CoulombMatrix', 'krr_ft']): 1024,
    frozenset(['qm8', 'CoulombMatrix', 'tf_regression_ft']): [26, 26],
    frozenset(['qm8', 'CoulombMatrix', 'dtnn']): [26, 26],
    frozenset(['qm8', 'GraphConv', 'graphconvreg']): 75,
    frozenset(['qm8', 'Weave', 'weave_regression']): 75,
    frozenset(['qm8', 'BPSymmetryFunction', 'ani']): [26, 4],
    frozenset(['qm8', 'MP', 'mpnn']): [70, 8],
    frozenset(['qm8', 'Raw', 'textcnn_regression']): None,

    frozenset(['qm9', 'ECFP', 'krr']): 1024,
    frozenset(['qm9', 'ECFP', 'rf_regression']): 1024,
    frozenset(['qm9', 'ECFP', 'tf_regression']): 1024,
    frozenset(['qm9', 'CoulombMatrix', 'krr_ft']): 1024,
    frozenset(['qm9', 'CoulombMatrix', 'tf_regression_ft']): [29, 29],
    frozenset(['qm9', 'CoulombMatrix', 'dtnn']): [29, 29],
    frozenset(['qm9', 'GraphConv', 'graphconvreg']): 75,
    frozenset(['qm9', 'Weave', 'weave_regression']): 75,
    frozenset(['qm9', 'BPSymmetryFunction', 'ani']): [29, 4],
    frozenset(['qm9', 'MP', 'mpnn']): [70, 8],
    frozenset(['qm9', 'Raw', 'textcnn_regression']): None,
  })
  tasks = dict({
    'qm7': None,
    'qm7b': [3, 4],
    'qm8': None,
    'qm9': ['homo', 'lumo'],
    'esol': None,
    'freesolv': None
  })
  splits = ['Random', 'Stratified']

  fracs = [float(x+1)/10 for x in range(8)]

  metrics = dict({
    'qm7': ['MAE'], # deepchem.metrics.Metric(deepchem.metrics.mae_score)
    'qm7b': ['MAE'],
    'qm8': ['MAE'],
    'qm9': ['MAE'],
    'esol': ['RMSE'],
    'freesolv': ['RMSE']
  })

  params = dict({
    'datasets': datasets,
    'featurizers': featurizers,
    'modes': modes,
    'methods': methods,
    'models': models,
    'features': features,
    'tasks': tasks,
    'splits': splits,
    'fracs': fracs,
    'metrics': metrics
  })

  # load default molnet hyper_parameters,
  # and evaluate on train and valid sets,
  # using valid score to optimize hyper_parameters using a gaussian process.
  #   - saving hyper_parameters via pickle
  benchmark(**params,
            hyper_parameters_init=None,
            hyper_parameter_search=True,
            valid=True,
            test=False,
            out_path=pathlib.Path('.') / 'benchmark' / 'optimization',
            load_hyper_parameters=False,
            save_hyper_parameters=True,
            save_results=False,
            reload=False,
            seed=SEED)

  # load optimized hyper_parameters,
  # and evaluate on train and test sets.
  #   - saving the results to a csv
  benchmark(**params,
            hyper_parameters_init=None,
            hyper_parameter_search=False,
            valid=False,
            test=True,
            out_path=pathlib.Path('.') / 'benchmark' / 'evaluation',
            load_hyper_parameters=True,
            save_hyper_parameters=False,
            save_results=True,
            reload=False,
            seed=SEED)

  # load the optimal hyper_parameters computed for the molnet/deepchem paper (via pickle),
  # and evaluate on train and test sets.
  #   - saving the results to a csv
  benchmark(**params,
            hyper_parameters_init=None,
            hyper_parameter_search=False,
            valid=False,
            test=True,
            out_path=pathlib.Path('.') / 'benchmark' / 'molnet',
            load_hyper_parameters=True,
            save_hyper_parameters=True,
            save_results=True,
            reload=False,
            seed=SEED)
  return None


if __name__ == '__main__':
  main()