from tensorflow.python.keras.metrics import categorical_accuracy
from models import ModelConfig

simple_encoder_classifier = ModelConfig(
                            epoch_num = 10, batch_num = 4,
                            # attention config
                            head_size=256,
                            num_heads=6,
                            ff_dim=12,
                            num_transformer_blocks = 2,
                            bias = True,
                            mh_drop = 0.15,
                            ff_drop = 0.25,

                            # mlp block
                            dlayers = [32],
                            d_kreg = None,
                            dense_drop = [0.3],
                            dense_init = ('RandomUniform', None),
                            dense_act = ['linear'],

                            # output
                            out_pooling = 'LAST',
                            out_type = 'categories',
                            out_act = 'sigmoid' ,
                            out_init = ('RandomUniform', None),

                            # compilation
                            optimizer = ('Adam', {'amsgrad': True}),
                            loss = 'binary_crossentropy',
                            metrics = ['binary_crossentropy', 'accuracy'],

                            # training
                            lr = 0.005,
                            shuffle = True,
                            early_stop =
                                         {'monitor':"val_loss",
                                          'min_delta':0.002,
                                          'patience':11,
                                          'verbose':1,
                                          'mode':"auto",
                                          'baseline': None,
                                          'restore_best_weights':False,
                                          'start_from_epoch': 16},
                            lr_drop = {'monitor':"val_loss",
                                          'factor':0.2,
                                          'patience':2,
                                          'verbose':1,
                                          'mode':"auto",
                                          'min_delta':0.0005,
                                          'cooldown':2,
                                          'min_lr':0.000001},
                            cv_split = 0.2,
                            pos_enc = ['sinecosine', None]
)

lstm_model = ModelConfig(
                            epoch_num = 10, batch_num = 4,
                            # attention config
                            head_size=None,
                            num_heads=2,
                            ff_dim=None,
                            num_transformer_blocks = None,
                            bias = None,
                            mh_drop = 0.15,
                            ff_drop = None,

                            # mlp block
                            dlayers = [32],
                            d_kreg = None,
                            dense_drop = [0.3],
                            dense_init = ('RandomUniform', None),
                            dense_act = ['linear'],

                            # output
                            out_pooling = None,
                            out_type = 'categories',
                            out_act = 'sigmoid' ,
                            out_init = ('RandomUniform', None),

                            # compilation
                            optimizer = ('Adam', {'amsgrad': True}),
                            loss = 'binary_crossentropy',
                            metrics = ['binary_crossentropy', 'accuracy'],

                            # training
                            lr = 0.005,
                            shuffle = True,
                            early_stop =
                                         {'monitor':"val_loss",
                                          'min_delta':0.002,
                                          'patience':11,
                                          'verbose':1,
                                          'mode':"auto",
                                          'baseline': None,
                                          'restore_best_weights':True,
                                          'start_from_epoch': 16},
                            lr_drop = {'monitor':"val_loss",
                                          'factor':0.2,
                                          'patience':2,
                                          'verbose':1,
                                          'mode':"auto",
                                          'min_delta':0.0005,
                                          'cooldown':2,
                                          'min_lr':0.000001},
                            cv_split = 0.03,
                            pos_enc = ['sinecosine', None]
)

def make_model_name_string(mtype:str, x_shape:tuple, model_parameters:ModelConfig)->str:
    return f'{mtype}_{x_shape[0]}_{x_shape[1]}_{x_shape[2]}_EP-{model_parameters.epoch_num}_MHA-' +\
                   f'{model_parameters.head_size}x{model_parameters.num_heads}-{model_parameters.ff_dim}_' +\
                   f'MLP-{model_parameters.dlayers}_Pool{model_parameters.out_pooling}_pos{model_parameters.pos_enc}'

def make_model_spec_string(l:list)->str:
    model_spec = f'Model Parameters:\n  ' + \
                 f'MLA:            {l[2:9]}\n  MLP:            {l[9:14]}\n  OUT:            {l[14:18]}' + \
                 f'\n  OPT:            {l[18:23]}\n  TRAIN:          {l[:2]},{l[23:24]}\n  TRAIN:           {l[24:25]}' + \
                 f'\n  TRAIN:           {l[25:]}'
    return model_spec
def make_experiment_spec_string(mtype:str, x_shape:tuple, ts_x_shape: tuple, model_parameters:ModelConfig)->str:
    return f'Experiment Parameters: model type: {mtype},seq_len: {x_shape[1]}, train_rows: {x_shape[0]}, test_rows: {ts_x_shape[0]}\n' +\
                      f'xcols: {x_shape[2]}\nPositional encodings: f{model_parameters.pos_enc}'