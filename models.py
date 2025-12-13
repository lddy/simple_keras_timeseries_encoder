import tensorflow as tf
import keras

from keras.models import Sequential
import keras.layers as layers

from keras.layers import Dense, LSTM
from keras.layers import LSTM, InputLayer, Bidirectional, Input, GRU
from keras.layers import Dropout
from keras.src.optimizers import Adam, SGD, RMSprop, Adagrad, Lion, AdamW,Adadelta,Adafactor, Nadam,Muon, Adamax
from keras.initializers import (RandomUniform, RandomNormal, TruncatedNormal,
                                Zeros, Ones,
                                GlorotNormal, GlorotUniform,
                                HeNormal, HeUniform,
                                Orthogonal,
                                Constant, VarianceScaling, Identity,
                                LecunNormal, LecunUniform)
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
import numpy as np
from collections import namedtuple
ModelConfig = namedtuple( 'ModelConfig',
[
    'epoch_num',
    'batch_num',
 #attention config
     'head_size', 'num_heads', 'ff_dim', 'bias', 'num_transformer_blocks', 'mh_drop', 'ff_drop',
 #ff block config
    'dlayers', 'd_kreg', 'dense_drop', 'dense_init', 'dense_act',
 #output config
    'out_pooling',
    'out_type',
    'out_act',
    'out_init',
 #model compilation
    'optimizer',
    'loss',
    'metrics',
 #training parameters
    'lr',
    'shuffle',
    'early_stop',
    'lr_drop',
    'cv_split',
    'pos_enc'
])
def transformer_encoder(inputs, head_size, num_heads, ff_dim, bias, mh_drop = 0, ff_drop = 0, useMHA = True):
    # Attention and Normalization
    if useMHA:
        if bias is None:
            bias = True
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=mh_drop, use_bias = bias
        )(query = inputs, value = inputs) #key is optional
        x = layers.Dropout(mh_drop)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        #residual bypass connection
        res = x + inputs
    else: #skip MHA
        res = inputs
    # Feed Forward Block and Normalization
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(ff_drop)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # residual bypass connection
    x = x + res
    return x

def build_transformer_encoder_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    bias,
    num_transformer_blocks,
    mh_drop,
    ff_drop,
    out_pooling,
    mlp_units,
    mlp_activation,
    n_classes,
    mlp_dropout,
    out_type,
    out_shape,
    out_act,
    out_init
    ):
    inputs = keras.Input(shape=input_shape)
    x = inputs

    #transformer block stack
    useMHA = True
    if input_shape[-2] == 1:
        useMHA = False
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, bias, mh_drop, ff_drop, useMHA)
    #reshape sequence to a single vector
    match out_pooling:
        case 'AVG':
            x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
        case 'MAX':
            x = layers.GlobalMaxPooling1D(data_format="channels_last")(x)
        case 'LAST':
            if input_shape[-2] > 1:
                x = layers.Cropping1D(cropping=(input_shape[-2]-1, 0))(x)
            x = layers.Flatten()(x)
        case 'LSTM':
            x = layers.LSTM(input_shape[-2])(x)
        case _:
            raise NotImplementedError(f'{out_pooling} is not implemented.')
    #output pre-processing block
    for i, dim in enumerate(mlp_units):
        x = layers.Dense(dim, activation=mlp_activation[i])(x)
        x = layers.Dropout(mlp_dropout[i])(x)
    if out_type == 'vect':
        outputs = layers.Dense(n_classes, activation=out_act, kernel_initializer=out_init)(x)
    else: #categories
        outputs = layers.Dense(out_shape, activation=out_act)(x)

    return keras.Model(inputs, outputs)

def pos_encoder_sinecosine(seq_len:int, d:int, n:int=10000):
    r = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            r[k, 2*i] = np.sin(k/denominator)
            r[k, 2*i+1] = np.cos(k/denominator)
    return r

def apply_pos_encoder(t:np.ndarray, pe:np.ndarray):
    for i in range(t.shape[0]):
        t[i] = t[i] + pe
    return t

def make_initializer(name:str, params:dict|None = None)->keras.initializers.Initializer:
    if params is None:
        params = {}
    init = eval(name)(**params)
    return init

def make_optimizer(name:str, params:dict|None = None, lr:float = 0.01)->keras.optimizers.Optimizer:
    if params is None:
        params = {'learning_rate': lr}
    else:
        params['learning_rate'] = lr
    opt = eval(name)(**params)
    return opt

def make_callback(name:str, params:dict)->Callback:
    if name == 'early_stop':
        if params is None:
            print('Warning: no parameters were specified for early stopping, using defaults')
            params = {'monitor':"val_loss",
                      'min_delta':0.0,
                      'patience':3,
                      'verbose':0,
                      'mode':"auto",
                      'baseline':None,
                      'restore_best_weights':False,
                      'start_from_epoch':20}
        return EarlyStopping(**params)
    if name == 'lr_drop':
        if params is None:
            print('Warning: no parameters were specified for learning rate reduction, using defaults')
            params = {'monitor':"val_loss",
                      'factor':0.1,
                      'patience':10,
                      'verbose':1,
                      'mode':"auto",
                      'min_delta':0.0001,
                      'cooldown':0,
                      'min_lr':0.0}
        return ReduceLROnPlateau(**params)
    raise NotImplementedError(f'make_callback: Unknown callback: ' + name)

def make_model_name_string(x_shape:tuple, model_parameters:ModelConfig)->str:
    return f'{x_shape[0]}_{x_shape[1]}_{x_shape[2]}_EP-{model_parameters.epoch_num}_MHA-' +\
                   f'{model_parameters.head_size}x{model_parameters.num_heads}-{model_parameters.ff_dim}_' +\
                   f'MLP-{model_parameters.dlayers}_Pool{model_parameters.out_pooling}_pos{model_parameters.pos_enc}'

def make_model_spec_string(l:list)->str:
    model_spec = f'Model Parameters:\n  ' + \
                 f'MLA:            {l[2:9]}\n  MLP:            {l[9:14]}\n  OUT:            {l[14:18]}' + \
                 f'\n  OPT:            {l[18:23]}\n  TRAIN:          {l[:2]},{l[23:24]}\n  TRAIN:           {l[24:25]}' + \
                 f'\n  TRAIN:           {l[25:]}'
    return model_spec
def make_experiment_spec_string(x_shape:tuple, ts_x_shape: tuple, model_parameters:ModelConfig)->str:
    return f'Experiment Parameters: seq_len: {x_shape[1]}, train_rows: {x_shape[0]}, test_rows: {ts_x_shape[0]}\n' +\
                      f'xcols: {x_shape[2]}\nPositional encodings: f{model_parameters.pos_enc}'
