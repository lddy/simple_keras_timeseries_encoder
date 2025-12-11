# simple_keras_timeseries_encoder
keras encoder MHA model example and test dataset

hat_model.py: data generation
generate_hat_states.py: execute data generation

models.py: MHA + FF attention block and encoder-only transformer definition + helper functions for training
model_definitions.py: parameters for the model and the training loop (edit to adjust model construction)

main.py: executes data shaping, model training and evaluation
