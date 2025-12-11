import tensorflow as tf
import keras

from keras.models import Sequential
import keras.layers as layers
from keras.layers import Dense
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
    'cv_split'
])
def transformer_encoder(inputs, head_size, num_heads, ff_dim, bias, mh_drop = 0, ff_drop = 0):
    # Attention and Normalization
    if bias is None:
        bias = True
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=mh_drop, use_bias = bias
    )(query = inputs, value = inputs) #key is optional
    x = layers.Dropout(mh_drop)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    #residual bypass connection
    res = x + inputs

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
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, bias, mh_drop, ff_drop)
    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)

    #output pre-processing block
    for i, dim in enumerate(mlp_units):
        x = layers.Dense(dim, activation=mlp_activation[i])(x)
        x = layers.Dropout(mlp_dropout[i])(x)
    if out_type == 'vect':
        outputs = layers.Dense(n_classes, activation=out_act, kernel_initializer=out_init)(x)
    else: #categories
        outputs = layers.Dense(out_shape, activation=out_act)(x)

    return keras.Model(inputs, outputs)

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

