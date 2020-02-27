import os
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
from datetime import datetime


def smape(true,predicted):
    """Symmetric mean absolute percentage error loss function
    
    :param true: true values
    :type true: np.array
    :param predicted: predicted values
    :type predicted: np.array
    :return: smape loss
    :rtype: float
    """    
    epsilon = 0.1
    summ = K.maximum(K.abs(true) + K.abs(predicted) + epsilon, 0.5 + epsilon)
    smape = K.abs(predicted - true) / summ * 2.0
    return smape


def model_fn_embedding(
        shape_x_1,
        shape_x_2,
        shape_y,
        n_units=250,
        output_n_layers=100,
        learning_rate=0.0004,
        dropout=0.1,
        return_estimator=False,
        conv_filters=64,
        conv_kernel_size=3,
        model_dir=None,
        loss=smape
):
    """Seq2Seq model for time series forecasting.

    :param shape_x_1: shape of x1. Variables in encoder.
    :type shape_x_1: tuple
    :param shape_x_2: shape of x2: exogenous variables inserted in decoder
    :type shape_x_2: tuple
    :param shape_y: shape of y
    :type shape_y: tuple
    :param n_units: number of units of LSTM's, defaults to 250
    :type n_units: int, optional
    :param output_n_layers: number of units of output hidden layer, defaults to 100
    :type output_n_layers: int, optional
    :param learning_rate: learning rate, defaults to 0.0004
    :type learning_rate: float, optional
    :param dropout: dropout for output hidden layer, defaults to 0.1
    :type dropout: float, optional
    :param return_estimator: wether you want to return an estimator or a keras model, defaults to False
    :type return_estimator: bool, optional
    :param conv_filters: The number of filters of the convolution, defaults to 64
    :type conv_filters: int, optional
    :param conv_kernel_size: the kernel size of the convolution, defaults to 3
    :type conv_kernel_size: int, optional
    :param model_dir: the model dir, defaults to None
    :type model_dir: str, optional
    :param loss: The loss function, defaults to 'mse'
    :type loss: str, optional
    :return: the compiled model.
    :rtype: py:class:`tensorflow.keras.Model`
    """

    encoder_inputs = tf.keras.layers.Input(
        shape=shape_x_1, name='Features_previous_days')

    conv = tf.keras.layers.Conv1D(
        filters=conv_filters,
        kernel_size=conv_kernel_size,
        strides=1,
        data_format='channels_last',
        input_shape=shape_x_1,
        activation='relu')(encoder_inputs)
    pool = tf.keras.layers.MaxPool1D(data_format='channels_last')(conv)

    encoder = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            n_units,
            return_state=True,
            name='Encoder'))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(
        pool)
    encoder_states = [forward_h, forward_c, backward_h, backward_c]

    decoder_inputs = tf.keras.layers.Input(
        shape=shape_x_2, name='Price_next_days')
    decoder_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            n_units,
            return_sequences=True,
            return_state=True,
            name='Decoder'))
    decoder_outputs, _, _, _, _ = decoder_lstm(
        decoder_inputs, initial_state=encoder_states)

    output_hidden_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(output_n_layers, activation='relu'))(decoder_outputs)
    output_hidden_layer = tf.keras.layers.Dropout(dropout)(output_hidden_layer)
    output_layer = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(shape_y[-1], activation='relu'))(output_hidden_layer)

    model = tf.keras.Model([encoder_inputs, decoder_inputs], output_layer)

    opt = tf.keras.optimizers.Adam(learning_rate)

    model.compile(loss=loss, optimizer=opt)

    if return_estimator:
        assert model_dir is not None, "Must define a gcs bucket directory"
        config = tf.estimator.RunConfig(
            model_dir=os.path.join(
                model_dir,
                "model",
                str(round(datetime.now().timestamp())) ),
            save_checkpoints_steps=600)
        estimator = tf.keras.estimator.model_to_estimator(
            keras_model=model, config=config)
        return estimator
    else:
        return model