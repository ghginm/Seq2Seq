import os
import sys
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine


## Load and split data

def get_data(db_url, data_path, sql_access=True):
    if sql_access:
        engine = create_engine(db_url)
        query = '''
                SELECT week, QuantityDal, ID FROM ml_data
                '''

        data = pd.read_sql(query, con=engine)
    else:
        with open(data_path, 'rb') as handle:
            data = pq.read_table(handle).to_pandas()

    return data

## Preprocess data

def split_data(data, date_col, test_start='2023-06-19', test_end=None):
    """Test / train splitting."""

    if test_end is None:
        data_test = data[data[date_col] >= test_start].reset_index(drop=True)
        data_train = data[data[date_col] < test_start].reset_index(drop=True)
    else:
        data_test = data[(data[date_col] >= test_start) & (data[date_col] <= test_end)].reset_index(drop=True)
        data_train = data[data[date_col] < test_start].reset_index(drop=True)

    return data_test, data_train


def preprocess_data(data, id_col, date_col, target_col, empty_token=0.0):
    """Data padding, recovering 0 sales, reordering."""

    if not (isinstance(id_col, str) and isinstance(date_col, str) and isinstance(target_col, str)):
        print('`id_col`, `date_col`, `target_col` must be of type string.')
        sys.exit(1)

    if len(data.columns) != 3:
        print('The dataset is expected to have 3 columns: `id_col`, `date_col`, `target_col`, which is necessary for'
              'correctly recovering 0s.')
        sys.exit(1)

    empty_token = float(empty_token)

    # Reordering columns
    data = data[[id_col, date_col, target_col]]

    # Replacing sales < 0 with 0
    data[target_col] = np.maximum(data[target_col], 0)

    # Recovering 0s
    data = data.set_index(list(set(data.columns) - {target_col}))[[target_col]].unstack().fillna(0).stack().reset_index()
    data['csum'] = data.groupby([id_col], observed=True)[target_col].cumsum()
    data[target_col] = [empty_token if x == 0 else y for x, y in zip(data['csum'], data[target_col])]
    data['sku_launch'] = [1 if x == 0 else 0 for x in data['csum']]
    data = data.drop('csum', axis=1).reset_index(drop=True)

    # Sorting data
    data = data.sort_values([date_col] + [id_col]).reset_index(drop=True)

    return data


def ts_var(data, id_col, date_col, target_col, date_freq: Literal['daily', 'weekly']='weekly',
           distant_lags=[51, 52, 53], sin_cos_vars=False, lag_vars=False, data_vars=None, empty_token=0):
    """Creating time-series variables: sine / cosine pairs, distant lags."""

    if not (isinstance(id_col, str) and isinstance(date_col, str) and isinstance(target_col, str)):
        print('`id_col`, `date_col`, `target_col` must be of type string.')
        sys.exit(1)

    if not isinstance(distant_lags, list):
        print('`distant_lags` must be a list of integers (lags).')
        sys.exit(1)

    data[date_col] = pd.to_datetime(data[date_col])

    if date_freq == 'daily':
        if sin_cos_vars:
            data['day'] = data[date_col].dt.day

            for i in [1, 2, 4, 52]:
                data[f'f_sin_52_{i}'] = np.sin((i * 2 * np.pi * data['day']) / 365)
                data[f'f_cos_52_{i}'] = np.cos((i * 2 * np.pi * data['day']) / 365)

            data = data.drop('day', axis=1)
    else:
        if sin_cos_vars:
            data['weekyear'] = [x.isocalendar()[1] for x in data[date_col]]

            for i in [1, 2, 12]:
                data[f'f_sin_52_{i}'] = np.sin((i * 2*np.pi * data['weekyear']) / 52)
                data[f'f_cos_52_{i}'] = np.cos((i * 2*np.pi * data['weekyear']) / 52)

            data = data.drop('weekyear', axis=1)

    if lag_vars:
        for i in distant_lags:
            data[f'lag_{i}'] = data.groupby([id_col], observed=True,
                                             group_keys=False)[target_col].shift(i).fillna(empty_token)

    if data_vars is not None:
        data = data.merge(data_vars, how='left', on=[id_col, date_col])

    return data

## Assess model performance

def model_evaluation(data, date_col, y_true, y_pred_list):
    data = data.copy()
    final_scores = []

    for i in data[date_col].unique():
        data_date = data[data[date_col] == i].copy()
        score_rmse, score_mae, score_fa, score_bias = [], [], [], []

        for j in y_pred_list:
            rmse = mean_squared_error(data_date[y_true], data_date[j], squared=False)
            score_rmse.append(round(rmse, 0))
            mae = mean_absolute_error(data_date[y_true], data_date[j])
            score_mae.append(round(mae, 0))

            data_date[f'delta_{j}'] = data_date[j] - data_date[y_true]
            data_date[f'delta_abs_{j}'] = np.abs(data_date[f'delta_{j}'])

        bias = [x for x in data_date.columns if 'delta' in x and 'abs' not in x]
        fa = [x for x in data_date.columns if 'delta_abs' in x]

        for fc, fa, bias in zip(y_pred_list, fa, bias):
            score_fa.append(round((1 - np.sum(data_date[fa]) / np.sum(data_date[fc]))*100, 2))
            score_bias.append(round((np.sum(data_date[bias]) / np.sum(data_date[fc]))*100, 2))

        df_scores = pd.DataFrame(zip(score_rmse, score_mae, score_fa, score_bias), index=y_pred_list,
                                 columns=['RMSE', 'MAE', 'Forecast accuracy, %', 'BIAS, %'])

        df_scores['date'] = i
        final_scores.append(df_scores)

    final_scores = pd.concat(final_scores)

    return final_scores

## Analyse validation results

def store_validation_versions(path_project, data, date_col, model_config, loss_tr_val):
    """Saving validation results."""

    # Creating dataframes for hyperparameters and loss
    version_date = data[date_col].max().strftime('%Y-%m-%d')

    version_hyperparam = pd.DataFrame(model_config, index=['value']).T.reset_index()
    version_hyperparam['value'] = version_hyperparam['value'].astype(str)
    version_loss = pd.DataFrame(loss_tr_val, index=['train_loss', 'val_loss']).T

    # Creating a unique file name
    file_validation = os.listdir(f'{path_project}\\validation\\')

    if len(file_validation) == 0:
        idx_hyperparam, idx_loss = 0, 0
    else:
        idx_hyperparam = [int(x.split('_')[0]) for x in file_validation if 'hyperparam' in x]
        idx_loss = [int(x.split('_')[0]) for x in file_validation if 'loss' in x]

    file_name_hyperparam = f'{np.max(idx_hyperparam) + 1}_{version_date}_hyperparam'
    file_name_loss = f'{np.max(idx_loss) + 1}_{version_date}_loss'

    # Saving the result
    with open(f'{path_project}\\validation\\{file_name_hyperparam}.parquet', 'wb') as handle:
        pq.write_table(pa.Table.from_pandas(version_hyperparam), handle, compression='GZIP')
    with open(f'{path_project}\\validation\\{file_name_loss}.parquet', 'wb') as handle:
        pq.write_table(pa.Table.from_pandas(version_loss), handle, compression='GZIP')


def collect_validation_versions(path_project):
    # Merging files
    file_validation = os.listdir(f'{path_project}\\validation\\')
    file_name_hyperparam = [x for x in file_validation if 'hyperparam' in x]
    file_name_loss = [x for x in file_validation if 'loss' in x]

    file_hyperparam, file_loss = [], []

    for i in file_name_hyperparam:
        file = pq.read_table(f'{path_project}\\validation\\{i}').to_pandas()
        file['version_idx'] = i.split('_')[0]
        file['version_date'] = i.split('_')[1]
        file_hyperparam.append(file)

    file_hyperparam = pd.concat(file_hyperparam)

    for i in file_name_loss:
        file = pq.read_table(f'{path_project}\\validation\\{i}').to_pandas()
        file['version_idx'] = i.split('_')[0]
        file['version_date'] = i.split('_')[1]
        file_loss.append(file)

    file_loss = pd.concat(file_loss)

    return file_hyperparam, file_loss


if __name__ == '__main__':
    pass