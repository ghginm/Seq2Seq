import random
import sys
from copy import deepcopy
from itertools import chain
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


## Preprocess data

class DatasetUtil(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    # Get the index
    def __getitem__(self, i):
        return self.x[i], self.y[i]


class DatasetTransform:
    """Transforms data for supervised learning, i.e. reshaping, scaling, creating DataLoaders.

    Attributes:
        data : pandas DataFrame.
        seq_len : the sliding window length, e.g. the rolling window of 90 days.
        fc_window : a forecast horizon.
        id_list : the list with all ids representing individual time series.
        id_n : the number of ids.
        id_col : the name of the id column.
        target_col : the name of the target columns.
        var_cols : a list with the names of all variables.
    """

    def __init__(self, data, seq_len, fc_window, id_list, id_n, id_col, target_col, var_cols):
        self.data = data.copy()
        self.seq_len = seq_len
        self.fc_window = fc_window
        self.id_list = id_list
        self.id_n = id_n
        self.id_col = id_col
        self.target_col = target_col
        self.var_cols = var_cols

        # Making sure that the target is the first column
        self.var_cols.insert(0, self.var_cols.pop(self.var_cols.index(self.target_col)))

        # Validating function parameters
        if not (isinstance(self.seq_len, int) and isinstance(self.fc_window, int) and
                isinstance(self.id_list, list) and isinstance(self.id_n, int) and
                isinstance(self.id_col, str) and isinstance(self.var_cols, list)):
            print(' `seq_len`, `fc_window`, `id_n` must be of type int. \n '
                  '`id_list`, `var_cols` must be lists. \n `id_col` must be of type string.')
            sys.exit(1)

    def id_shape(self, col_vector, inference=False):
        """Reshaping time series, using the sliding window approach.

        Example:
            The first column is the target variable (Y), the rest are exogenous variables (X):

            [[22, 1, 0], [33, 2, 0], [44, 3, 0],
             [55, 4, 0], [44, 3, 0], [66, 5, 0]]

            Transforming the dataset to a supervised problem assuming the `seq_len` = 2 and `fc_window` = 2:

            X_1 = np.array([[22, 1, 0], [33, 2, 0]]), Y_1 = [44, 55]
            X_2 = np.array([[33, 2, 0], [44, 3, 0]]), Y_2 = [55, 66]
        """

        # Validating function parameters
        if self.fc_window < 2:
            print('The value for `fc_window` must be >= 2.')
            sys.exit(1)

        # TODO: change `seq_len` during inference?

        if inference:
            x, y = [], None

            window = col_vector[-self.seq_len:]  # `seq_len` is arbitrary during inference
            x.append(window)
            x = np.array(x, dtype=np.float32)
        else:
            seq_n = len(col_vector) - (self.seq_len + self.fc_window - 1)  # the number of sliding windows the target will be broken into
            x = np.zeros([seq_n, self.seq_len, col_vector.shape[1]], dtype=np.float32)
            y = np.zeros([seq_n, self.fc_window, 1], dtype=np.float32)  # for now, keep only the target (no other variables)

            for i in range(seq_n):
                x[i, :, :] = col_vector[i:(i + self.seq_len), :]
                y[i, :, :] = col_vector[(i + self.seq_len):(i + self.seq_len + self.fc_window), 0].reshape(-1, 1)

        return x, y

    def id_shape_all(self, id_list, inference=False):
        """Iterating through each id and applying `ts_shape()`."""

        if inference:
            x, y = [], None
            # Before filtering, data should be sorted by 'date', 'id'
            data_filtered = self.data[-(self.seq_len * self.id_n):].copy()  # `seq_len` is arbitrary during inference
            for i in id_list:
                var_id = data_filtered.loc[data_filtered[self.id_col] == i, self.var_cols].to_numpy()
                x_id, _ = self.id_shape(col_vector=var_id, inference=True)
                x.append(x_id)
            x = np.concatenate(x)
            del data_filtered
        else:
            x, y = [], []

            for i in id_list:
                var_id = self.data.loc[self.data[self.id_col] == i, self.var_cols].to_numpy()
                x_id, y_id = self.id_shape(col_vector=var_id, inference=False)
                x.append(x_id)
                y.append(y_id)
            x, y = np.concatenate(x), np.concatenate(y)

        return x, y

    def val_idx(self, val_size, data_transformed=True):
        """Produces indices to split data into train / validation set."""

        if data_transformed:
            # After using the sliding-window approach (`id_shape_all()`), the dataset gets reduced
            idx_len = self.data.shape[0] - (self.seq_len + self.fc_window - 1) * self.id_n
            len_padded = idx_len / self.id_n
            idx_val, remove = [], []

            # Getting validation indices, taking into account hierarchical structure
            for i in range(idx_len + 1):
                if (i != 0) and (i % len_padded == 0):
                    idx_val.append(list(range((i - val_size), i)))
                    # Removing the intersection between train and validation set
                    remove.append(list(range((i - val_size), (i - val_size + self.fc_window - 1))))

            idx_val, remove = list(chain.from_iterable(idx_val)), list(chain.from_iterable(remove))
            idx_tr = list(set(range(idx_len)) - set(idx_val))
            idx_val = list(set(idx_val) - set(remove))
        else:
            idx_len = self.data.shape[0]
            idx_data = range(idx_len)
            idx_val = None  # list(idx_data[-val_size * self.id_n:])
            idx_tr = list(idx_data[:-val_size * self.id_n])

        return idx_val, idx_tr

    def scale_fit(self, sk_scaler, idx_tr):
        sk_scaler.fit(self.data.loc[idx_tr, self.var_cols].to_numpy())

        return sk_scaler

    def prepare_data(self, val_size, sk_scaler, batch_size,
                     operational_mode: Literal['validation', 'training', 'inference']='training'):
        """Prepares data for different operational modes, i.e. reshaping, scaling, creating DataLoaders.

        Parameters:
            val_size : the number of sequences of length `seq_len` in the validation set.
            sk_scaler : sklearn's scaler.
            batch_size : the size of a batch.
            operational_mode : a mode that defines which operations to apply.
        """

        if operational_mode == 'validation':
            # Getting train / validation indices
            idx_val_scaling, idx_tr_scaling = self.val_idx(val_size=val_size, data_transformed=False)
            idx_val, idx_tr = self.val_idx(val_size=val_size, data_transformed=True)

            # Scaling data
            scaler = self.scale_fit(sk_scaler=sk_scaler, idx_tr=idx_tr_scaling)
            self.data[self.var_cols] = scaler.transform(self.data[self.var_cols].to_numpy())
            self.scaler = scaler  # !!

            # Applying the sliding-window approach
            x, y = self.id_shape_all(id_list=self.id_list, inference=False)

            # Data loaders
            x_tr, y_tr = x[idx_tr], y[idx_tr]
            x_val, y_val = x[idx_val], y[idx_val]
            del x, y

            loader_tr = DataLoader(DatasetUtil(x=x_tr, y=y_tr), batch_size=batch_size, shuffle=True)
            loader_val = DataLoader(DatasetUtil(x=x_val, y=y_val), batch_size=batch_size, shuffle=False)
            del x_tr, y_tr, x_val, y_val
        elif operational_mode == 'training':
            # Scaling data
            scaler = self.scale_fit(sk_scaler=sk_scaler, idx_tr=self.data.index)
            self.data[self.var_cols] = scaler.transform(self.data[self.var_cols].to_numpy())
            self.scaler = scaler  # !!

            # Applying the sliding-window approach
            x, y = self.id_shape_all(id_list=self.id_list, inference=False)

            loader_tr = DataLoader(DatasetUtil(x=x, y=y), batch_size=batch_size, shuffle=True)
            loader_val = None
            del x, y
        else:
            # Scaling data
            scaler = self.scale_fit(sk_scaler=sk_scaler, idx_tr=self.data.index)
            self.data[self.var_cols] = scaler.transform(self.data[self.var_cols].to_numpy())
            self.scaler = scaler  # !!

            # Applying the sliding-window approach (NO BATCHING)
            x, y = self.id_shape_all(id_list=self.id_list, inference=True)

            loader_tr = torch.from_numpy(x)  # loader_tr = DataLoader(DatasetUtil(x=x, y=y), batch_size=batch_size, shuffle=True)
            loader_val = None
            del x, y

        return loader_tr, loader_val

## Build a model

class EncoderLSTM(nn.Module):
    """Creates an encoder LSTM NN.

    Attributes:
        input_size : the number of features.
        hidden_size : the number of neurons, i.e. LSTM cells.
        n_layer : the number of stacked layers.
        dropout_prob : zeroing out the output of a random unit.
    """

    def __init__(self, input_size, hidden_size, n_layer, dropout_prob, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.dropout_prob = dropout_prob
        self.device = device

        # Validating function parameters
        if (self.dropout_prob > 0) and (self.n_layer == 1):
            print('If `n_layer` == 1, `dropout_prob` must be equal to 0.')
            sys.exit(1)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.n_layer,
                            dropout=self.dropout_prob, batch_first=True)

    def forward(self, x):
        # Initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.n_layer, batch_size, self.hidden_size).to(self.device)

        # Forward pass
        _, hc_state = self.lstm(x, (h0, c0))

        return hc_state


class DecoderLSTM(nn.Module):
    """Creates an encoder LSTM NN.

    Attributes:
        hidden_size_dec : the number of neurons in the decoder.
        hidden_size_enc : the number of neurons in the encoder.
        n_layer_dec : the number of stacked layers in the decoder.
        n_layer_enc : the number of stacked layers in the encoder.
        dropout_prob : zeroing out the output of a random unit.
        hc_multilayer_stack : determines whether only the hidden state of the final layer should be kept
        or the hidden states from all layers should be reshaped and passed to the decoder.
    """

    def __init__(self, hidden_size_dec, hidden_size_enc, n_layer_dec, n_layer_enc,
                 dropout_prob, device, hc_multilayer_stack=False):
        super().__init__()
        self.input_size = 1
        self.hidden_size_dec = hidden_size_dec
        self.hidden_size_enc = hidden_size_enc
        self.n_layer_dec = n_layer_dec
        self.n_layer_enc = n_layer_enc
        self.dropout_prob = dropout_prob
        self.device = device
        self.hc_multilayer_stack = hc_multilayer_stack

        self.match_hidden_state_final_layer = nn.Linear(self.hidden_size_enc, self.hidden_size_dec)
        self.match_hidden_state_multi_layer = nn.Linear((self.n_layer_enc * self.hidden_size_enc), self.hidden_size_dec)

        # Validating function parameters
        if (self.dropout_prob > 0) and (self.n_layer_dec == 1):
            print('If `n_layer_dec` == 1, `dropout_prob` should be equal to 0.')
            sys.exit(1)

        self.lstm = nn.LSTM(self.input_size, self.hidden_size_dec, self.n_layer_dec,
                            dropout=self.dropout_prob, batch_first=True)

        self.output_layer = nn.Linear(self.hidden_size_dec, 1)  # a fully connected layer (output)

    def forward(self, x, hc_state_enc, decoder_inner_loop):
        # The encoder-decoder transition (1st iteration)
        if decoder_inner_loop == 0:
            # Reshaping hidden states of the encoder:
            if self.n_layer_enc > 1:
                # To disrupt the gradient flow between the encoder and the decoder, use .detach() on the hidden / cell states
                if self.hc_multilayer_stack:
                    # Stacking the encoder's hidden states: (n_layer, batch_size, hidden_size) => (1, batch_size, n_layer*hidden_size)
                    hidden_state_enc = hc_state_enc[0].view(1, -1, (self.n_layer_enc * self.hidden_size_enc))  # .detach()
                    cell_state_enc = hc_state_enc[1].view(1, -1, (self.n_layer_enc * self.hidden_size_enc))  # .detach()
                    hc_state_dec = (hidden_state_enc, cell_state_enc)
                else:
                    # Getting the last hidden state of the encoder: (n_layer, batch_size, hidden_size) => (1, batch_size, hidden_size).
                    # The first layer of the decoder's hidden state is initialised with the last one of the encoder
                    hidden_state_enc = hc_state_enc[0][-1, :, :].unsqueeze(0)  # .detach().unsqueeze(0)
                    cell_state_enc = hc_state_enc[1][-1, :, :].unsqueeze(0)  # .detach().unsqueeze(0)
                    hc_state_dec = (hidden_state_enc, cell_state_enc)
            else:
                hc_state_dec = hc_state_enc

            # If the decoder's hidden state has a different shape than the encoder's one, use a linear layer
            if self.hc_multilayer_stack:
                hidden_state_dec = self.match_hidden_state_multi_layer(hc_state_dec[0])
                cell_state_dec = self.match_hidden_state_multi_layer(hc_state_dec[1])
                hc_state_dec = (hidden_state_dec, cell_state_dec)
            elif self.hidden_size_dec != self.hidden_size_enc:
                hidden_state_dec = self.match_hidden_state_final_layer(hc_state_dec[0])
                cell_state_dec = self.match_hidden_state_final_layer(hc_state_dec[1])
                hc_state_dec = (hidden_state_dec, cell_state_dec)

            # If the decoder has more than 1 layer, initialise all other layers with 0
            if self.n_layer_dec > 1:
                batch_size = x.size(0)
                h0 = torch.zeros((self.n_layer_dec - 1), batch_size, self.hidden_size_dec).to(self.device)
                c0 = torch.zeros((self.n_layer_dec - 1), batch_size, self.hidden_size_dec).to(self.device)
                hc_state_dec = (torch.cat((hc_state_dec[0], h0), dim=0), torch.cat((hc_state_dec[1], c0), dim=0))
        else:
            hc_state_dec = hc_state_enc

        # Forward pass
        lstm_output, hc_state = self.lstm(x, hc_state_dec)  # dimensions of `lstm_output`: (batch_size, seq_len, hidden_size)
        output = self.output_layer(lstm_output[:, -1, :])

        return output, hc_state


class EncoderDecoderLSTM(nn.Module):
    """Creates an encoder-decoder LSTM NN."""

    def __init__(self, device, fc_window_train, fc_window_inference, input_size_enc, hidden_size_enc, n_layer_enc,
                 dropout_prob_enc, hidden_size_dec, n_layer_dec, dropout_prob_dec, hc_multilayer_stack=False):
        super().__init__()

        # Shared parameters
        self.device = device
        self.fc_window_train = fc_window_train
        self.fc_window_inference = fc_window_inference

        # Encoder's parameters
        self.input_size_enc = input_size_enc
        self.hidden_size_enc = hidden_size_enc
        self.n_layer_enc = n_layer_enc
        self.dropout_prob_enc = dropout_prob_enc

        # Decoder's parameters
        self.hidden_size_dec = hidden_size_dec
        self.n_layer_dec = n_layer_dec
        self.dropout_prob_dec = dropout_prob_dec
        self.hc_multilayer_stack = hc_multilayer_stack

        # Instantiating the encoder and decoder
        self.encoder = EncoderLSTM(input_size=self.input_size_enc, hidden_size=self.hidden_size_enc, n_layer=self.n_layer_enc,
                                   dropout_prob=self.dropout_prob_enc, device=self.device)

        self.decoder = DecoderLSTM(hidden_size_dec=self.hidden_size_dec, hidden_size_enc=self.hidden_size_enc,
                                   n_layer_dec=self.n_layer_dec, n_layer_enc=self.n_layer_enc,
                                   dropout_prob=self.dropout_prob_dec, device=self.device,
                                   hc_multilayer_stack=self.hc_multilayer_stack)

    def forward(self, x, y, operational_mode: Literal['training', 'validation', 'inference']='training',
                training_mode: Literal['recursive', 'teacher-forcing', 'mixed']='mixed', teacher_forcing_prob=0.0):
        # Forward pass (encoder)
        encoder_hc_state = self.encoder(x)  # a callable instance of the `EncoderLSTM` class

        # Transforming input to predict the 1st time step (decoder): (batch_size, seq_len, input_size) => (batch_size, 1, 1)
        decoder_input = x[:, -1, 0].unsqueeze(1).unsqueeze(1)
        decoder_hc_state = encoder_hc_state

        # Forward pass (decoder)
        if operational_mode == 'training':
            # Collecting predictions
            decoder_batch_output = torch.zeros(x.size(0), self.fc_window_train, 1)

            if training_mode == 'recursive':
                for i in range(self.fc_window_train):
                    decoder_output, decoder_hc_state = self.decoder(decoder_input, decoder_hc_state, i)
                    decoder_batch_output[:, i, :] = decoder_output
                    decoder_input = decoder_output.unsqueeze(1)  # (batch_size, 1, 1)

            if training_mode == 'teacher-forcing':
                for i in range(self.fc_window_train):
                    decoder_output, decoder_hc_state = self.decoder(decoder_input, decoder_hc_state, i)
                    decoder_batch_output[:, i, :] = decoder_output
                    decoder_input = y[:, i, :].unsqueeze(1)  # (batch_size, 1, 1)

            if training_mode == 'mixed':
                for i in range(self.fc_window_train):
                    decoder_output, decoder_hc_state = self.decoder(decoder_input, decoder_hc_state, i)
                    decoder_batch_output[:, i, :] = decoder_output

                    # Randomly choosing training strategies (bernoulli trial)
                    bernoulli_rv = [1 if random.random() < teacher_forcing_prob else 0][0]
                    if bernoulli_rv == 1:
                        decoder_input = y[:, i, :].unsqueeze(1)  # (batch_size, 1, 1)
                    else:
                        decoder_input = decoder_output.unsqueeze(1)  # (batch_size, 1, 1)
        elif operational_mode == 'validation':
            # Collecting predictions
            decoder_batch_output = torch.zeros(x.size(0), self.fc_window_train, 1)

            # Validation can be only recursive
            for i in range(self.fc_window_train):
                decoder_output, decoder_hc_state = self.decoder(decoder_input, decoder_hc_state, i)
                decoder_batch_output[:, i, :] = decoder_output
                decoder_input = decoder_output.unsqueeze(1)  # (batch_size, 1, 1)
        else:
            # Collecting predictions (NO BATCHING)
            decoder_batch_output = torch.zeros(x.shape[0], self.fc_window_inference, 1)

            # Validation / inference can be only recursive
            for i in range(self.fc_window_inference):
                decoder_output, decoder_hc_state = self.decoder(decoder_input, decoder_hc_state, i)
                decoder_batch_output[:, i, :] = decoder_output
                decoder_input = decoder_output.unsqueeze(1)  # (batch_size, 1, 1)

        return decoder_batch_output

## Train a model and forecast

def train_validate(model, device, dataloader_train, dataloader_val, n_epoch, l_rate, weight_decay=0,
                   training_mode: Literal['recursive', 'teacher-forcing', 'mixed']='mixed', teacher_forcing_prob=0.5,
                   validate=False, store_last_param=0, samp_freq_param=1, ma_weights=False, beta=0.65):
    """Runs the training / validation loop.

    Parameters:
        model : a trained encoder-decoder LSTM NN.
        dataloader_train : DataLoader for training (use `prepare_data()`).
        dataloader_val : DataLoader for validation (use `prepare_data()`).
        n_epoch : the number of epochs for training.
        l_rate : a learning rate.
        weight_decay : L2 regularisation.
        training_mode : whether to train a model recursively, through teacher-forcing or both.
        teacher_forcing_prob : the probability to use teacher forcing for the mixed strategy (bernoulli rv)
        validate : whether to do both training and validation.
        store_last_param : for how many final epochs parameters should be stored.
        samp_freq_param : the frequency of storing parameters, i.e. storing parameters every n epochs.
        ma_weights : if True, weights are replaced with the exponential moving average.
        beta : the smoothing factor.
    """

    # Validating function parameters
    if store_last_param >= n_epoch:
        print('Make sure that `store_last_param` < `n_epoch`.')
        exit(1)

    # Optimizer, loss, parameters
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()
    loss_training, loss_validation = [], []
    param_epoch, param_avg = [], {}

    # Checkpoints
    if not validate:
        last_epochs = np.array(range(n_epoch))[-store_last_param:] + 1
        checkpoints = last_epochs[last_epochs % samp_freq_param == 0]

        print(f'With `store_last_param` = {store_last_param} and `samp_freq_param` = {samp_freq_param} '
              f'a total of {len(checkpoints)} models will be saved. \n'
              f'Epochs: {checkpoints[0:4]} ... {checkpoints[-4:]}')

    for epoch in range(n_epoch):
        # Training
        model.train()
        loss_train_epoch_cumsum = float(0)
        params_epoch = {}

        for idx, batch in enumerate(dataloader_train):
            # Batch dimensions: (batch_size, seq_len, input_size), (batch_size, fc_window, input_size)
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass (encoder-decoder)
            output = model(x_batch, y_batch, operational_mode='training', training_mode=training_mode,
                           teacher_forcing_prob=teacher_forcing_prob)

            # Loss
            loss_batch = loss_function(output.to(device), y_batch)
            loss_train_epoch_cumsum += loss_batch.item()

            # Backward pass
            loss_batch.backward()
            optimizer.step()

        # Average training loss
        loss_train_epoch_avg = loss_train_epoch_cumsum / len(dataloader_train)
        loss_training.append(loss_train_epoch_avg)
        
        # Calculating MA of weights
        if ma_weights and (store_last_param > 0):
            # Storing MA of parameters
            if epoch > int(0.1*n_epoch):
                if not param_avg:
                    for key, value in model.named_parameters():
                        param_avg[key] = value.data.clone()  # value.detach().clone()
                else:
                    for key, value in model.named_parameters():
                        param_avg[key] = beta * param_avg[key] + (1 - beta) * value.data

            # Storing last n parameters
            if (((epoch + 1) >= n_epoch - store_last_param) and ((epoch + 1) % samp_freq_param == 0) and (store_last_param > 0)):
                for key, value in param_avg.items():
                    params_epoch[key] = value
                param_epoch.append(deepcopy(params_epoch))
        else:
            # Storing last n parameters
            if (((epoch + 1) >= n_epoch - store_last_param) and ((epoch + 1) % samp_freq_param == 0) and (store_last_param > 0)):
                for key, value in model.named_parameters():
                    params_epoch[key] = value
                param_epoch.append(deepcopy(params_epoch))

        # Validation
        if validate:
            model.eval()
            loss_val_epoch_cumsum = float(0)

            for idx, batch in enumerate(dataloader_val):
                # Batch dimensions: (batch_size, seq_len, input_size), (batch_size, fc_window, input_size)
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)

                with torch.no_grad():
                    # Forward pass (encoder-decoder)
                    output = model(x_batch, y_batch, operational_mode='validation', training_mode=training_mode,
                                   teacher_forcing_prob=teacher_forcing_prob)

                    # Loss
                    loss_batch = loss_function(output.to(device), y_batch)
                    loss_val_epoch_cumsum += loss_batch.item()

            # Average validation loss
            loss_val_epoch_avg = loss_val_epoch_cumsum / len(dataloader_val)
            loss_validation.append(loss_val_epoch_avg)

            # Printing results
            print(f'Epoch {epoch + 1} / {n_epoch} | training loss: {np.round(loss_train_epoch_avg, 8)} | '
                  f'validation loss: {np.round(loss_val_epoch_avg, 8)}')
        else:
            loss_validation = None
            print(f'Epoch {epoch + 1} / {n_epoch} | training loss: {np.round(loss_train_epoch_avg, 8)}')

    return (loss_training, loss_validation), param_epoch


def forecast(model, device, data_pred):
    """Produces forecasts."""
    model.to(device)
    forecast = model(data_pred.to(device), torch.empty(0), operational_mode='inference')
    forecast = forecast.detach().cpu().numpy().flatten()

    return forecast


def avg_param(param_list):
    param_avg = dict.fromkeys(param_list[0].keys(), 0)
    n_avg = len(param_list)

    for i in range(n_avg):
        params_epoch = param_list[i]
        for key, value in params_epoch.items():
            param_avg[key] += (1 / n_avg) * value.detach()

    return param_avg


if __name__ == '__main__':
    pass
