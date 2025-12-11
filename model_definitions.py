from tensorflow.python.keras.metrics import categorical_accuracy
from models import ModelConfig

simple_encoder_classifier = ModelConfig(
                            epoch_num = 35, batch_num = 20,
                            # attention config
                            head_size=256,
                            num_heads=6,
                            ff_dim=None,
                            num_transformer_blocks = 3,
                            bias = True,
                            mh_drop = 0.2,
                            ff_drop = 0.2,

                            # mlp block
                            dlayers = [32],
                            d_kreg = None,
                            dense_drop = [0.3],
                            dense_init = ('RandomUniform', None),
                            dense_act = ['linear'],

                            # output
                            out_type = 'categories',
                            out_act = 'sigmoid' ,
                            out_init = ('RandomUniform', None),

                            # compilation
                            optimizer = ('Adam', {'amsgrad': True}),
                            loss = 'binary_crossentropy',
                            metrics = ['binary_crossentropy', 'accuracy'],

                            # training
                            lr = 0.004,
                            shuffle = True,
                            early_stop = None,
                                         # {'monitor':"val_loss",
                                         #  'min_delta':0.0002,
                                         #  'patience':9,
                                         #  'verbose':1,
                                         #  'mode':"auto",
                                         #  'baseline': None,
                                         #  'restore_best_weights':True,
                                         #  'start_from_epoch': 20},
                            lr_drop = {'monitor':"val_loss",
                                          'factor':0.2,
                                          'patience':4,
                                          'verbose':1,
                                          'mode':"auto",
                                          'min_delta':0.001,
                                          'cooldown':4,
                                          'min_lr':0.000001},
                            cv_split = 0.0)
