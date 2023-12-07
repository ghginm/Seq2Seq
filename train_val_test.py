import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

import data_processing as dp
import model_creation as mc


## Initial parameters

path_project = str(Path('__file__').absolute().parent)

# Config
with open(f'{path_project}\\config.json', 'r') as config_file:
    config = json.load(config_file)

model_config, model_train_val_test = config['model_configuration'], config['model_train_val_test']

# Modes
validation_mode = model_train_val_test['validation_mode']
val_size = model_train_val_test['val_size']  # the number of sequences of length `seq_len` in the validation set.

training_mode = model_train_val_test['training_mode']
train_n_models = model_train_val_test['train_n_models']

testing_mode = model_train_val_test['testing_mode']
get_test_results = model_train_val_test['get_test_results']

## Data (train, test)

id_col, date_col, target_col = 'id', 'date', 'target'
fc_window = model_config['fc_window']

data = dp.get_data(db_url='mysql+mysqlconnector:...', data_path=f'{path_project}\\data\\data.parquet', sql_access=False)
data = dp.preprocess_data(data=data, date_col=date_col, id_col=id_col, target_col=target_col, empty_token=0)

if testing_mode:
    test_start = data['date'].max() - pd.to_timedelta(fc_window - 1, unit='W')
    data_test, data = dp.split_data(data=data, date_col=date_col, test_start=test_start, test_end=None)

## Variables

data = dp.ts_var(data=data, id_col=id_col, date_col=date_col, target_col=target_col,
                 date_freq='weekly', distant_lags=[51, 52, 53],
                 sin_cos_vars=True, lag_vars=True, data_vars=None, empty_token=0)

## Script parameters

# Parameters
input_col = [col for col in data.columns if col not in {'id', 'date'}]  # Columns must have the same order during training and inference
# list(set(data.columns) - {'id', 'date'})

id_list = list(data['id'].unique())
id_n = len(id_list)

sk_scaler = MinMaxScaler()

# Hyperparameters
seq_len = model_config['seq_len']

l_rate = model_config['l_rate']
dropout_prob_enc = model_config['dropout_prob_enc']
dropout_prob_dec = model_config['dropout_prob_dec']
weight_decay = model_config['weight_decay']

input_size_enc = len(input_col)
hidden_size_enc = model_config['hidden_size_enc']
hidden_size_dec = model_config['hidden_size_dec']
n_layer_enc = model_config['n_layer_enc']
n_layer_dec = model_config['n_layer_dec']

batch_size = model_config['batch_size']
n_epoch_validate = model_config['n_epoch_validate']
n_epoch_train = model_config['n_epoch_train']
store_last_param = model_config['store_last_param']
samp_freq_param = model_config['samp_freq_param']

training_strategy = model_config['training_strategy']
teacher_forcing_prob = model_config['teacher_forcing_prob']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Validation

if validation_mode:
    print('Validation')

    data_transform = mc.DatasetTransform(data=data, seq_len=seq_len, fc_window=fc_window, id_list=id_list,
                                         id_n=id_n, id_col=id_col, target_col=target_col, var_cols=input_col)

    loader_tr, loader_val = data_transform.prepare_data(val_size=val_size, sk_scaler=sk_scaler,
                                                        batch_size=batch_size, operational_mode='validation')

    model = mc.EncoderDecoderLSTM(device=device, fc_window_train=fc_window, fc_window_inference=None,
                                  input_size_enc=input_size_enc, hidden_size_enc=hidden_size_enc,
                                  n_layer_enc=n_layer_enc, dropout_prob_enc=dropout_prob_enc,
                                  hidden_size_dec=hidden_size_dec, n_layer_dec=n_layer_dec,
                                  dropout_prob_dec=dropout_prob_dec, hc_multilayer_stack=False)

    loss_tr_val, _ = mc.train_validate(model=model, device=device, dataloader_train=loader_tr, dataloader_val=loader_val,
                                       n_epoch=n_epoch_validate, l_rate=l_rate, weight_decay=weight_decay,
                                       training_mode=training_strategy, teacher_forcing_prob=teacher_forcing_prob,
                                       validate=True, store_last_param=0)

    # Saving validation results (including all previous versions)
    dp.store_validation_versions(path_project=path_project, data=data, date_col=date_col,
                                 model_config=model_config, loss_tr_val=loss_tr_val)

    results_val = dp.collect_validation_versions(path_project=path_project)
    with pd.ExcelWriter(f'{path_project}\\validation\\validation_results.xlsx', engine='xlsxwriter') as writer:
        results_val[0].to_excel(writer, sheet_name='Sheet1', index=False)
        results_val[1].to_excel(writer, sheet_name='Sheet2', index=False)

## Training multiple models (`train_n_models` models)

if training_mode:
    print('Training mode: on')

    for train_iter in range(train_n_models):
        print(f'Model {train_iter}')

        # Preparing data
        data_transform = mc.DatasetTransform(data=data, seq_len=seq_len, fc_window=fc_window, id_list=id_list,
                                             id_n=id_n, id_col=id_col, target_col=target_col, var_cols=input_col)

        loader_tr, _ = data_transform.prepare_data(val_size=val_size, sk_scaler=sk_scaler,
                                                   batch_size=batch_size, operational_mode='training')

        # Initialising a model
        model = mc.EncoderDecoderLSTM(device=device, fc_window_train=fc_window, fc_window_inference=None,
                                      input_size_enc=input_size_enc, hidden_size_enc=hidden_size_enc,
                                      n_layer_enc=n_layer_enc, dropout_prob_enc=dropout_prob_enc,
                                      hidden_size_dec=hidden_size_dec, n_layer_dec=n_layer_dec,
                                      dropout_prob_dec=dropout_prob_dec, hc_multilayer_stack=False)

        # Training
        loss_tr, param = mc.train_validate(model=model, device=device, dataloader_train=loader_tr, dataloader_val=None,
                                           n_epoch=n_epoch_train, l_rate=l_rate, weight_decay=weight_decay,
                                           training_mode=training_strategy, teacher_forcing_prob=teacher_forcing_prob,
                                           validate=False, store_last_param=store_last_param, samp_freq_param=samp_freq_param)

        # Saving parameters
        if len(param) > 0:
            torch.save(param, f'{path_project}\\model\\model_{train_iter}_param_n_epoch.pth')
            # param_avg_epoch = mc.avg_param(param_list=param)
        else:
            torch.save(model.state_dict(), f'{path_project}\\model\\model_{train_iter}_param_epoch.pth')

    # Saving forecast horizon and the order of columns
    hyperparam_inference = {'fc_window': fc_window, 'input_col': data_transform.var_cols}

    with open(f'{path_project}\\model\\hyperparam_inference.json', 'w') as json_file:
        json.dump(hyperparam_inference, json_file)

## Testing

if testing_mode and get_test_results:
    print('Testing mode: on')

    # Comparing forecast horizon and the order of columns. Training vs inference
    with open(f'{path_project}\\model\\hyperparam_inference.json', 'r') as config_file:
        hyperparam_inference = json.load(config_file)

    if not (hyperparam_inference['fc_window'] == fc_window) and (hyperparam_inference['input_col'] == input_col):
        print('Ensure that `fc_window` and `input_col` used during inference are the same as during training.')
        sys.exit(1)

    # Preparing data
    data_transform = mc.DatasetTransform(data=data, seq_len=seq_len, fc_window=fc_window, id_list=id_list,
                                         id_n=id_n, id_col=id_col, target_col=target_col, var_cols=input_col)

    data_pred, _ = data_transform.prepare_data(val_size=val_size, sk_scaler=sk_scaler,
                                               batch_size=batch_size, operational_mode='inference')

    # Forecasting (n models, each has m checkpoints)
    fc_avg = []
    model_param_all = os.listdir(f'{path_project}\\model\\')
    model_param_all = [x for x in model_param_all if '.pth' in x]

    for idx, model_param in enumerate(model_param_all):
        print(f'Model {idx}')

        # Loading parameters
        checkpoint_param = torch.load(f'{path_project}\\model\\{model_param}')

        # Checkpoints
        for idx, param in enumerate(checkpoint_param):
            print(f'Checkpoint {idx}')

            model = mc.EncoderDecoderLSTM(device=device, fc_window_train=fc_window, fc_window_inference=fc_window,
                                          input_size_enc=input_size_enc, hidden_size_enc=hidden_size_enc,
                                          n_layer_enc=n_layer_enc, dropout_prob_enc=dropout_prob_enc,
                                          hidden_size_dec=hidden_size_dec, n_layer_dec=n_layer_dec,
                                          dropout_prob_dec=dropout_prob_dec, hc_multilayer_stack=False)

            model.load_state_dict(param)
            model.eval()

            # Forecasting
            forecast = mc.forecast(model=model, device=device, data_pred=data_pred)

            # Scaling data back
            dummy_matrix = np.zeros((forecast.shape[0], len(input_col) - 1), dtype=np.float32)
            forecast_rescaled = np.column_stack((forecast, dummy_matrix))
            forecast_rescaled = data_transform.scaler.inverse_transform(forecast_rescaled)[:, 0]
            forecast_rescaled[forecast_rescaled < 0] = 0

            fc_avg.append(forecast_rescaled)

    fc_avg = np.average(fc_avg, axis=0)

    # Evaluating model performance
    date_list = list(pd.date_range(start=test_start, periods=fc_window, freq='W-MON'))
    id_date_list = [x for x in id_list for _ in range(fc_window)]
    fc = pd.DataFrame([id_date_list, len(id_list) * date_list], index=[id_col, date_col]).T
    fc['fc_avg'] = fc_avg
    data_test_pred = data_test.merge(fc, how='left', on=[id_col, date_col])

    metrics = dp.model_evaluation(data=data_test_pred, date_col='date', y_true='target', y_pred_list=['fc_avg'])