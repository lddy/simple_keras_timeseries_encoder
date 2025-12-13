import os

from models import *
from model_definitions import *
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model
from datetime import datetime
import misc_utils
from timeit import default_timer as timer

rebuild_sequences = False
test_only = False # when True, the model will be loaded from file, instead of being (re)fitted
continue_training = False

model_type = 'transformer-encoder'
data_file = 'weather_data_with_forecasts_and_hat_state.csv'
cached_sequence_X_y_data = 'sequence_data.jl'
cached_model = 'trained_model.keras'

seq_len = 30
xcols = ['MOY', 'new_hat', 'HolidayTomorrow', 'WeekendTomorrow', 'picnic', 'is_sick',
         'Temp', 'Temp_NextForecast', 'Prec_NextForecast', 'Precipitation',
          ]
ycols = ['hat']

naive_pred_cols = ['naive_prediction']
train_rows = 14000
test_rows = 2500
# ^ must ^ run ^ with ^ rebuild_sequences == True ^ at ^ least ^ once ^ or
# ^ after ^ changing ^ any ^ of ^ the ^ above ^ parameters ^



model_parameters = simple_encoder_classifier
named_params = list(model_parameters._asdict().items())

def shape_data(df: pd.DataFrame,
               xcols: list, ycols: list,
               seq_len:int,
               train_rows:int = 2000,
               logger:misc_utils.simple_echo_logger = None):
    pred_cols = []
    for yc in ycols:
        df[f'{yc}_next'] = df[yc].shift(-1)
        pred_cols.append(f'{yc}_next')

    df = df.dropna()
    rp_indexer = df[pred_cols].copy()
    rp_indexer['seq_elems'] = [[] for x in range(len(rp_indexer))]
    rp_indexer['sequence_len'] = 0
    tr_valid_idxlist = df.index.to_numpy()
    for i in rp_indexer.index:
        max_val_idx = i
        while max_val_idx not in tr_valid_idxlist and max_val_idx > 0:
            max_val_idx -= 1
        min_val_idx = max_val_idx
        while min_val_idx > 0 and len(tr_valid_idxlist[min_val_idx:max_val_idx + 1]) < seq_len:
            min_val_idx -= 1
        if min_val_idx < 0:
            continue
        elements = tr_valid_idxlist[min_val_idx:max_val_idx + 1].tolist()
        rp_indexer.at[i, 'seq_elems'] = elements
        rp_indexer.at[i, 'sequence_len'] = len(elements)
    first_full_seq_idx = rp_indexer.loc[rp_indexer['sequence_len'] == seq_len].index.min()
    start_idx = first_full_seq_idx - seq_len + 1
    df = df.iloc[start_idx:, :]
    rp_indexer = rp_indexer.loc[rp_indexer['sequence_len'] == seq_len]

    stacked_x = None
    stacked_y = None
    if logger is not None: logger.out(f'Collecting {len(rp_indexer)} input sequences')

    for i in rp_indexer.index:
        if i % 200 == 0:
            print(f'\t... processed {i} out of {len(rp_indexer)}...')

        curr_seq = df.loc[df.index.isin(rp_indexer['seq_elems'].at[i]), xcols + ycols].to_numpy()
        curr_r = df.loc[i:i, pred_cols].to_numpy()
        if stacked_x is None:
            stacked_x = curr_seq[None]
            stacked_y = curr_r[None]
        else:
            stacked_x = np.vstack((stacked_x, curr_seq[None]))
            stacked_y = np.vstack((stacked_y, curr_r[None]))

    if logger is not None: logger.out(f'Sequences ready, {train_rows} train_rows, {test_rows} test_rows\n')

    train = {}
    test = {}
    train['X'] = stacked_x[:train_rows]
    train['y'] = stacked_y[:train_rows]
    test['X'] = stacked_x[train_rows:]
    test['y'] = stacked_y[train_rows:]


    return df, rp_indexer, train, test

def configure_model(mc:ModelConfig, x_shape, y_shape):
    opt = make_optimizer(mc.optimizer[0], mc.optimizer[1], mc.lr)
    out_init = make_initializer(mc.out_init[0],mc.out_init[1])

    model = build_transformer_encoder_model(
        x_shape[1:],
        head_size=mc.head_size,
        num_heads=mc.num_heads,
        ff_dim=mc.ff_dim if mc.ff_dim is not None else x_shape[2],
        bias=mc.bias,
        mh_drop=mc.mh_drop,
        ff_drop=mc.ff_drop,
        out_pooling = mc.out_pooling,
        num_transformer_blocks=mc.num_transformer_blocks,
        mlp_units=mc.dlayers,
        mlp_activation=mc.dense_act,
        mlp_dropout=mc.dense_drop,
        out_type=mc.out_type,
        n_classes= y_shape[-1],
        out_shape= y_shape[-1],
        out_act=mc.out_act,
        out_init=out_init
    )

    model.compile(optimizer=opt, loss=mc.loss, metrics=mc.metrics)
    model.summary(expand_nested=True)
    return model


def fit_model(model,
              mc: ModelConfig,
              data: dict | None = None,
              y_flatten = True):

    x_data = data['X']
    y_data = data['y']

    y_shape = y_data.shape
    if y_flatten:
        y_data = y_data.reshape(y_shape[0], 1)
        y_shape = y_data.shape

    callbacks = []

    if mc.early_stop is not None:
        if mc.cv_split is None or mc.cv_split == 0.0:
            mc.early_stop['monitor'] = 'loss'
        es_monitor = make_callback(name = 'early_stop', params = mc.early_stop)
        callbacks.append(es_monitor)
    if mc.lr_drop is not None:
        if mc.cv_split is None or mc.cv_split == 0.0:
            mc.lr_drop['monitor'] = 'loss'
        lr_monitor = make_callback(name = 'lr_drop', params = mc.lr_drop)
        callbacks.append(lr_monitor)

    fit_hist = model.fit(x_data, y_data,
                         shuffle=mc.shuffle,
                         epochs=mc.epoch_num,
                         callbacks=callbacks,
                         batch_size=mc.batch_num,
                         validation_split=mc.cv_split)

    return model, fit_hist

def evaluate_model(model, data, y_flatten = True, batches = -1):
    x_data = data['X']
    y_data = data['y']

    y_shape = y_data.shape
    if y_flatten:
        y_data = y_data.reshape(y_shape[0], )
        y_shape = y_data.shape
    if batches > 0:
        predictions = model.predict(x_data, batch_size=64, verbose=1)
    else:
        predictions = model.predict(x_data, batch_size=1, verbose=1)
    return {'actual': y_data, 'predictions': predictions}

if __name__ == "__main__":
    lg = misc_utils.simple_echo_logger('outputlog.txt', append=False)

    time_data_shaping = None
    time_training = None
    time_inference = None

    if rebuild_sequences:
        lg.out('rebuilding sequences')
        data = pd.read_csv(data_file)

        data = data[xcols + ycols + naive_pred_cols]
        data = data.iloc[:train_rows + test_rows, :]
        t1 = timer()
        clean_data, resp_indexer, train_data, test_data = \
            shape_data(df = data, xcols = xcols, ycols= ycols, seq_len = seq_len, train_rows=train_rows, logger=lg)
        t2 = timer()
        time_data_shaping = np.round(t2 - t1, 3)
        sequence_data ={
            'clean_data':clean_data,
            'resp_indexer':resp_indexer,
            'train_data':train_data,
            'test_data':test_data
        }
        with open(cached_sequence_X_y_data, "wb") as jlfile:
            # noinspection PyTypeChecker
            joblib.dump(sequence_data, jlfile)
    else:
        with open(cached_sequence_X_y_data, "rb") as jlfile:
            sequence_data = joblib.load(jlfile)
        clean_data, resp_indexer, train_data, test_data =\
            sequence_data['clean_data'], sequence_data['resp_indexer'], sequence_data['train_data'], sequence_data['test_data']

    x_shape = train_data['X'].shape

    tr_x_unstacked = train_data['X'].reshape(x_shape[0]*x_shape[1], x_shape[2])
    scaler = MinMaxScaler()
    scaler.fit(X = tr_x_unstacked)
    tr_x_unstacked_scaled = scaler.transform(tr_x_unstacked)
    train_data['X'] = tr_x_unstacked_scaled.reshape(x_shape)

    ts_x_shape = test_data['X'].shape
    tst_x_unstacked = test_data['X'].reshape(ts_x_shape[0] * ts_x_shape[1], ts_x_shape[2])
    tst_x_unstacked_scaled = scaler.transform(tst_x_unstacked)
    test_data['X'] = tst_x_unstacked_scaled.reshape(ts_x_shape)

    lg.out('Scaled the data')

    if model_parameters.pos_enc[0] is not None and model_parameters.pos_enc[0] == 'sinecosine':
        pe = pos_encoder_sinecosine(seq_len = seq_len, d = x_shape[2])
        train_data['X'] = apply_pos_encoder(train_data['X'], pe)
        test_data['X'] = apply_pos_encoder(test_data['X'], pe)
        lg.out(f'Applied positional encoder ({model_parameters.pos_enc[0]})')


    model_string = make_model_name_string(x_shape=x_shape, model_parameters=model_parameters)
    figure_name = f'{model_string}.png'
    model_spec = make_model_spec_string(l=named_params)
    experiment_spec = make_experiment_spec_string(x_shape=x_shape, ts_x_shape=ts_x_shape, model_parameters=model_parameters)

    lg.out(f'model_string: {model_string}')
    lg.out(model_spec)

    if continue_training or test_only:
        m = keras.models.load_model(cached_model)
    else:
        m = configure_model(mc=model_parameters,
                            x_shape=train_data['X'].shape,
                            y_shape=(train_data['y'].shape[0], train_data['y'].shape[2]))  # "flatten" response vectors
    if not test_only:
        t1 = timer()
        m, fit_hist = fit_model(model = m,
                                mc=model_parameters,
                                data = train_data)
        m.save(cached_model)
        t2 = timer()
        time_training = np.round(t2 - t1, 3)

    m.summary()
    plot_model(m,
               to_file=figure_name,
               rankdir='BT',
               show_layer_activations=True,
               show_shapes=True,
               show_layer_names=True,
               dpi = 600)

    t1 = timer()
    results = evaluate_model(m, test_data, y_flatten = False)
    t2 = timer()
    time_inference = np.round(t2 - t1, 3)

    clean_data_trtsonly = clean_data.iloc[resp_indexer.index,:]
    results_pd = clean_data_trtsonly.iloc[x_shape[0]:,:].copy()
    if len(results_pd) != results['predictions'].shape[0]:
        lg.out(f'Warning: different test dataset results in data generation vs eval results:' +
               f' {len(results_pd)} vs {results["predictions"].shape[0]}', level=1)
    results_pd['pred'] = results['predictions'].reshape(results['predictions'].shape[0]).tolist()
    results_pd['actual'] = results['actual'].reshape(results['actual'].shape[0]).tolist()
    results_pd['pred_q'] = results_pd['pred'].apply(lambda x: 0 if x <0.5 else 1)
    results_pd.to_csv('experiment_result.csv')

    lg.out('\nModel-based predictions:')
    lg.out('confusion matrix:')
    tn, fp, fn, tp = confusion_matrix(results_pd['actual'].to_numpy(),
                                      results_pd['pred_q'].to_numpy(),
                                      normalize = 'all').ravel().tolist()
    cm_model = f'\tTN: {round(tn,2):.2f}\t FP: {round(fp,2):.2f}\n\tFN: {round(fn,2):.2f}\t TP: {round(tp,2):.2f}'
    lg.out(cm_model)
    acc_model = round(100*(tn + tp), 1)
    prec_model, rec_model = round(100* (tp / (tp + fp + 0.0000001)), 1), round(100* (tp / (tp + fn + 0.0000001)), 1)
    lg.out(f'accuracy: {acc_model}, precision: {prec_model}, recall: {rec_model}')

    lg.out('\nNaive Predictor results:')
    lg.out('confusion matrix')
    tn, fp, fn, tp = confusion_matrix(results_pd['hat_next'].to_numpy(),
                                      results_pd[naive_pred_cols[0]].to_numpy(),
                                      normalize='all').ravel().tolist()
    cm_naive = f'\tTN: {round(tn, 2):.2f}\t FP: {round(fp, 2):.2f}\n\tFN: {round(fn, 2):.2f}\t TP: {round(tp, 2):.2f}'
    lg.out(cm_naive)
    acc_naive = round(100 * (tn + tp), 1)
    prec_naive, rec_naive = round(100 * (tp / (tp + fp + 0.0000001)), 1), round(100 * (tp / (tp + fn + 0.0000001)), 1)
    lg.out(f'accuracy: {acc_naive}, precision: {prec_naive}, recall: {rec_naive}\n')

    model_log_dir = f'./history/{datetime.now().strftime("%m-%d-%y")}/t{datetime.now().strftime("%H%M")}_{model_string}'
    os.makedirs(model_log_dir, exist_ok = True)
    exp_filename = f'a{acc_model}_p{prec_model}_r{rec_model}.txt'

    exp_summ = misc_utils.simple_echo_logger(f'{model_log_dir}/{exp_filename}')
    exp_summ.out(f'{model_type}: {model_string}')
    exp_summ.out(model_spec)
    exp_summ.out(experiment_spec)

    exp_summ.out('-------------------------------')
    exp_summ.out(f'Model results:')
    exp_summ.out(cm_model)
    exp_summ.out(f'accuracy: {acc_model}, precision: {prec_model}, recall: {rec_model}')
    exp_summ.out('-------------------------------')
    exp_summ.out(f'Naive Predictor results:')
    exp_summ.out(cm_naive)
    exp_summ.out(f'accuracy: {acc_naive}, precision: {prec_naive}, recall: {rec_naive}\n')

    exp_summ.out(f'Data shaping time: {time_data_shaping}, Training time: {time_training}, Inference time: {time_inference}')
    exp_summ.stop()
    lg.out(f'Data shaping time: {time_data_shaping}, Training time: {time_training}, Inference time: {time_inference}')
    lg.out('Done')
    lg.stop()