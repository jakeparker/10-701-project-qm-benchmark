from __future__ import division
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import deepchem
from deepchem.molnet.preset_hyper_parameters import hps
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge

def train_test_benchmark_regression(train_dataset,
                                    test_dataset,
                                    tasks,
                                    transformers,
                                    n_features,
                                    metric,
                                    model,
                                    hyper_parameters=None,
                                    seed=123):
  """
  """
  train_scores = {}
  test_scores = {}

  assert model in [
      'tf_regression', 'tf_regression_ft', 'rf_regression', 'graphconvreg',
      'dtnn', 'dag_regression', 'xgb_regression', 'weave_regression',
      'textcnn_regression', 'krr', 'ani', 'krr_ft', 'mpnn'
  ]
  import xgboost
  if hyper_parameters is None:
    hyper_parameters = hps[model]
  model_name = model

  if model_name == 'tf_regression':
    layer_sizes = hyper_parameters['layer_sizes']
    weight_init_stddevs = hyper_parameters['weight_init_stddevs']
    bias_init_consts = hyper_parameters['bias_init_consts']
    dropouts = hyper_parameters['dropouts']
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']

    model = deepchem.models.MultiTaskRegressor(
        len(tasks),
        n_features,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        dropouts=dropouts,
        weight_decay_penalty=penalty,
        weight_decay_penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed)

  elif model_name == 'tf_regression_ft':
    layer_sizes = hyper_parameters['layer_sizes']
    weight_init_stddevs = hyper_parameters['weight_init_stddevs']
    bias_init_consts = hyper_parameters['bias_init_consts']
    dropouts = hyper_parameters['dropouts']
    penalty = hyper_parameters['penalty']
    penalty_type = hyper_parameters['penalty_type']
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    fit_transformers = [hyper_parameters['fit_transformers'](train_dataset)]

    model = deepchem.models.MultiTaskFitTransformRegressor(
        n_tasks=len(tasks),
        n_features=n_features,
        layer_sizes=layer_sizes,
        weight_init_stddevs=weight_init_stddevs,
        bias_init_consts=bias_init_consts,
        dropouts=dropouts,
        weight_decay_penalty=penalty,
        weight_decay_penalty_type=penalty_type,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fit_transformers=fit_transformers,
        n_eval=10,
        seed=seed)

  elif model_name == 'graphconvreg':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_filters = hyper_parameters['n_filters']
    n_fully_connected_nodes = hyper_parameters['n_fully_connected_nodes']

    model = deepchem.models.GraphConvModel(
        len(tasks),
        graph_conv_layers=[n_filters] * 2,
        dense_layer_size=n_fully_connected_nodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=seed,
        mode='regression')

  elif model_name == 'dtnn':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_embedding = hyper_parameters['n_embedding']
    n_distance = hyper_parameters['n_distance']
    assert len(n_features) == 2, 'DTNN is only applicable to qm datasets'

    model = deepchem.models.DTNNModel(
        len(tasks),
        n_embedding=n_embedding,
        n_distance=n_distance,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=seed,
        output_activation=False,
        use_queue=False,
        mode='regression')

  elif model_name == 'dag_regression':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    default_max_atoms = hyper_parameters['default_max_atoms']

    max_atoms_train = max([mol.get_num_atoms() for mol in train_dataset.X])
    max_atoms_test = max([mol.get_num_atoms() for mol in test_dataset.X])
    max_atoms = max([max_atoms_train, max_atoms_test])
    max_atoms = min([max_atoms, default_max_atoms])
    print('Maximum number of atoms: %i' % max_atoms)
    reshard_size = 256
    transformer = deepchem.trans.DAGTransformer(max_atoms=max_atoms)
    train_dataset.reshard(reshard_size)
    train_dataset = transformer.transform(train_dataset)
    test_dataset.reshard(reshard_size)
    test_dataset = transformer.transform(test_dataset)

    model = deepchem.models.DAGModel(
        len(tasks),
        max_atoms=max_atoms,
        n_atom_feat=n_features,
        n_graph_feat=n_graph_feat,
        n_outputs=30,
        batch_size=batch_size,
        learning_rate=learning_rate,
        random_seed=seed,
        use_queue=False,
        mode='regression')

  elif model_name == 'weave_regression':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_graph_feat = hyper_parameters['n_graph_feat']
    n_pair_feat = hyper_parameters['n_pair_feat']

    model = deepchem.models.WeaveModel(
        len(tasks),
        n_atom_feat=n_features,
        n_pair_feat=n_pair_feat,
        n_hidden=50,
        n_graph_feat=n_graph_feat,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        random_seed=seed,
        mode='regression')

  elif model_name == 'textcnn_regression':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    n_embedding = hyper_parameters['n_embedding']
    filter_sizes = hyper_parameters['filter_sizes']
    num_filters = hyper_parameters['num_filters']

    char_dict, length = deepchem.models.TextCNNModel.build_char_dict(
        train_dataset)

    model = deepchem.models.TextCNNModel(
        len(tasks),
        char_dict,
        seq_length=length,
        n_embedding=n_embedding,
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_queue=False,
        random_seed=seed,
        mode='regression')

  elif model_name == 'ani':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    layer_structures = hyper_parameters['layer_structures']

    assert len(n_features) == 2, 'ANI is only applicable to qm datasets'
    max_atoms = n_features[0]
    atom_number_cases = np.unique(
        np.concatenate([
            train_dataset.X[:, :, 0],
            test_dataset.X[:, :, 0]
        ]))

    atom_number_cases = atom_number_cases.astype(int).tolist()
    try:
      # Remove token for paddings
      atom_number_cases.remove(0)
    except:
      pass
    ANItransformer = deepchem.trans.ANITransformer(
        max_atoms=max_atoms, atom_cases=atom_number_cases)
    train_dataset = ANItransformer.transform(train_dataset)
    test_dataset = ANItransformer.transform(test_dataset)
    n_feat = ANItransformer.get_num_feats() - 1

    model = deepchem.models.ANIRegression(
        len(tasks),
        max_atoms,
        n_feat,
        layer_structures=layer_structures,
        atom_number_cases=atom_number_cases,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        mode="regression",
        random_seed=seed)

  elif model_name == 'mpnn':
    batch_size = hyper_parameters['batch_size']
    nb_epoch = hyper_parameters['nb_epoch']
    learning_rate = hyper_parameters['learning_rate']
    T = hyper_parameters['T']
    M = hyper_parameters['M']

    model = deepchem.models.MPNNModel(
        len(tasks),
        n_atom_feat=n_features[0],
        n_pair_feat=n_features[1],
        n_hidden=n_features[0],
        T=T,
        M=M,
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_queue=False,
        mode="regression")

  elif model_name == 'rf_regression':
    n_estimators = hyper_parameters['n_estimators']
    nb_epoch = None

    # Building scikit random forest model
    def model_builder(model_dir_rf_regression):
      sklearn_model = RandomForestRegressor(
          n_estimators=n_estimators, n_jobs=-1)
      return deepchem.models.sklearn_models.SklearnModel(
          sklearn_model, model_dir_rf_regression)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  elif model_name == 'krr':
    alpha = hyper_parameters['alpha']
    nb_epoch = None

    # Building scikit learn Kernel Ridge Regression model
    def model_builder(model_dir_krr):
      sklearn_model = KernelRidge(kernel="rbf", alpha=alpha)
      return deepchem.models.SklearnModel(sklearn_model, model_dir_krr)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  elif model_name == 'krr_ft':
    alpha = hyper_parameters['alpha']
    nb_epoch = None

    ft_transformer = deepchem.trans.CoulombFitTransformer(train_dataset)
    train_dataset = ft_transformer.transform(train_dataset)
    test_dataset = ft_transformer.transform(test_dataset)

    # Building scikit learn Kernel Ridge Regression model
    def model_builder(model_dir_krr):
      sklearn_model = KernelRidge(kernel="rbf", alpha=alpha)
      return deepchem.models.SklearnModel(sklearn_model, model_dir_krr)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  elif model_name == 'xgb_regression':
    max_depth = hyper_parameters['max_depth']
    learning_rate = hyper_parameters['learning_rate']
    n_estimators = hyper_parameters['n_estimators']
    gamma = hyper_parameters['gamma']
    min_child_weight = hyper_parameters['min_child_weight']
    max_delta_step = hyper_parameters['max_delta_step']
    subsample = hyper_parameters['subsample']
    colsample_bytree = hyper_parameters['colsample_bytree']
    colsample_bylevel = hyper_parameters['colsample_bylevel']
    reg_alpha = hyper_parameters['reg_alpha']
    reg_lambda = hyper_parameters['reg_lambda']
    scale_pos_weight = hyper_parameters['scale_pos_weight']
    base_score = hyper_parameters['base_score']
    seed = hyper_parameters['seed']
    early_stopping_rounds = hyper_parameters['early_stopping_rounds']
    nb_epoch = None

    esr = {'early_stopping_rounds': early_stopping_rounds}

    # Building xgboost regression model
    def model_builder(model_dir_xgb):
      xgboost_model = xgboost.XGBRegressor(
          max_depth=max_depth,
          learning_rate=learning_rate,
          n_estimators=n_estimators,
          gamma=gamma,
          min_child_weight=min_child_weight,
          max_delta_step=max_delta_step,
          subsample=subsample,
          colsample_bytree=colsample_bytree,
          colsample_bylevel=colsample_bylevel,
          reg_alpha=reg_alpha,
          reg_lambda=reg_lambda,
          scale_pos_weight=scale_pos_weight,
          base_score=base_score,
          seed=seed)
      return deepchem.models.xgboost_models.XGBoostModel(
          xgboost_model, model_dir_xgb, **esr)

    model = deepchem.models.multitask.SingletaskToMultitask(
        tasks, model_builder)

  print('-----------------------------')
  print('Start fitting: %s' % model_name)
  if nb_epoch is None:
    model.fit(train_dataset)
  else:
    model.fit(train_dataset, nb_epoch=nb_epoch)

  train_scores[model_name] = model.evaluate(train_dataset, metric, transformers)
  test_scores[model_name] = model.evaluate(test_dataset, metric, transformers)

  return train_scores, test_scores