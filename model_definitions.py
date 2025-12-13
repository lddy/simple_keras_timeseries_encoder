from tensorflow.python.keras.metrics import categorical_accuracy
from models import ModelConfig

simple_encoder_classifier = ModelConfig(
                            epoch_num = 30, batch_num = 4,
                            # attention config
                            head_size=256,
                            num_heads=6,
                            ff_dim=14,
                            num_transformer_blocks = 3,
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
                            out_pooling = 'LSTM',
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
