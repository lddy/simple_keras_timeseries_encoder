from models import *
from model_definitions import *
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from keras.utils import plot_model

data_file = 'weather_data_with_forecasts_and_hat_state.csv'
cached_sequence_X_y_data = 'sequence_data.jl'
cached_model = 'trained_model.keras'

seq_len = 30
xcols = ['new_hat', 'HolidayTomorrow', 'Temp_NextForecast', 'Prec_NextForecast', 'WeekendTomorrow']
ycols = ['hat']
naive_pred_cols = ['naive_prediction']
train_rows = 14000
test_rows = 4000
# ^ must ^ run ^ with ^ rebuild_sequences == True ^ at ^ least ^ once ^ or
# ^ after ^ changing ^ any ^ of ^ the ^ above ^ parameters ^

rebuild_sequences = True

# when True, the model will be loaded from file, instead of being re-fitted
test_only = False

model_parameters = simple_encoder_classifier

l = list(model_parameters._asdict().items())

print(f'Model Parameters:\n  MLA:            {l[2:9]}\n  MLP:            {l[9:14]}\n  OUT:            {l[14:17]}' +
      f'\n  OPT:            {l[17:21]}\n  TRAIN:          {l[:2]}, {l[21:]}')

def shape_data(df: pd.DataFrame,
               xcols: list, ycols: list,
               seq_len:int,
               train_rows:int = 2000):
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
    print(f'Collecting {len(rp_indexer)} input sequences')
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
    print(f'Sequences ready, {train_rows} train_rows, {test_rows} test_rows\n')
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
    figure_name = f'model_shape{x_shape[0]}-{x_shape[1]}_{x_shape[2]}_MHA-{mc.head_size}x{mc.num_heads}-{mc.ff_dim}_MLP-{mc.dlayers}.png'
    model.compile(optimizer=opt, loss=mc.loss, metrics=mc.metrics)
    model.summary()
    plot_model(model, to_file=figure_name, show_shapes=True, show_layer_names=True)
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
    if rebuild_sequences:
        data = pd.read_csv(data_file)

        data = data[xcols + ycols + naive_pred_cols]
        data = data.iloc[:train_rows + test_rows, :]

        clean_data, resp_indexer, train_data, test_data = \
            shape_data(df = data, xcols = xcols, ycols= ycols, seq_len = seq_len, train_rows=train_rows)

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

    print('Scaled the data')

    if not test_only:
        m = configure_model(mc = model_parameters,
                            x_shape = train_data['X'].shape,
                            y_shape = (train_data['y'].shape[0], train_data['y'].shape[2]) ) #"flatten" response vectors
        trained_model, fit_hist = fit_model(model = m,
                                            mc=model_parameters,
                                            data = train_data)
        trained_model.save(cached_model)
    else:
        trained_model = keras.models.load_model(cached_model)

    results = evaluate_model(trained_model, test_data, y_flatten = False)
    results_pd = clean_data.iloc[-ts_x_shape[0]:,:].copy()
    results_pd['pred'] = results['predictions'][:, 0].tolist()
    results_pd['pred_q'] = results_pd['pred'].apply(lambda x: 0 if x <0.5 else 1)

    print('\nModel-based predictions:')
    print('confusion matrix')
    tn, fp, fn, tp = confusion_matrix(results_pd['hat'].to_numpy(),
                                      results_pd['pred_q'].to_numpy(),
                                      normalize = 'all').ravel().tolist()
    print(f'\tTN: {round(tn,2):.2f}\t FP: {round(fp,2):.2f}\n\tFN: {round(fn,2):.2f}\t TP: {round(tp,2):.2f}')
    print(f'accuracy: {round(tn + tp, 2)}')

    print('\nNaive predictions:')
    print('confusion matrix')
    tn, fp, fn, tp = confusion_matrix(results_pd['hat'].to_numpy(),
                                      results_pd[naive_pred_cols[0]].to_numpy(),
                                      normalize='all').ravel().tolist()
    print(f'TN: {round(tn,2):.2f}\t FP: {round(fp,2):.2f}\t\nFN: {round(fn,2):.2f}\t TP: {round(tp,2):.2f}')
    print(f'accuracy: {round(tn + tp, 2)}\n\n')

    results_pd.to_csv('experiment_result.csv')

    print('Done')