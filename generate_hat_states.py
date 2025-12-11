from hat_model import neighbour
import pandas as pd
import numpy as np

dataset_output = './weather_data_with_forecasts_and_hat_state.csv'
diag_output = './intermediate_data/weather_data_with_forecasts_and_hat_state_diag.csv'
df = pd.read_csv('./intermediate_data/weather_data_with_forecasts.csv', index_col=0)


n = neighbour(df)
#simulate full timeline
n.run_all()

n.data.to_csv(dataset_output)
n.data_diag.to_csv(diag_output)
n.data['naive_prediction_error'] = n.data.apply(lambda row: np.abs(row.naive_prediction - row.hat), axis=1)
print(f'Global naive pred. accuracy: {round(1.0-n.data['naive_prediction_error'].sum()/len(n.data), 4)}')
print(f'Generated the dataset {dataset_output}, shape: {n.data.shape}, columns: {str(list(n.data.columns))}')

