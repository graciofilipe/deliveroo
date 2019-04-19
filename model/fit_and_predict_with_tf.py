import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from collections import Counter

import matplotlib.pyplot as plt

import tensorflow as tf


import plotly as py
import plotly.graph_objs as go

from sklearn.preprocessing import OneHotEncoder

class PrintDot(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 10 == 0: print('')
    print('.', end='')

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


import numpy as np
np.random.seed(42)


def model_data_preparation(feature_names, outcome_name):
    data = pd.read_csv('processed_data/order_and_restaurant_filtered_and_enriched.csv')
    data.drop(labels='Unnamed: 0', axis='columns', inplace=True)

    X = data[feature_names]
    y = data[outcome_name]

    return X, y

def data_prep(X):
    numeric_features = ['order_value', 'number_of_items',
                        'hour_order_acknowledged_at',
                        'rest_median_order_value',
                        'rest_median_number_of_items',
                        'rest_median_value_per_item']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['city', 'type_of_food', 'day_of_order', 'restaurant_id']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    X_transformed = preprocessor.fit_transform(X)

    return X_transformed

def build_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(44, activation=tf.nn.relu, input_shape=[input_shape]),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(22, activation=tf.nn.relu),
        tf.keras.layers.Dense(1)
        ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def predictions_pipeline():

    feature_names = ['order_value', 'number_of_items', 'hour_order_acknowledged_at',
                     'city', 'type_of_food', 'day_of_order', 'restaurant_id',
                     'rest_median_order_value', 'rest_median_number_of_items',
                     'rest_median_value_per_item']

    outcome_name = ['minutes_from_acknowledged_to_ready']


    X, y = model_data_preparation(feature_names, outcome_name)
    X_tr = data_prep(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tr, y, test_size=0.2)

    model = build_model(X_train.shape[1])
    print(model.summary())

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model.fit(X_train, y_train, epochs=100,
              validation_split=0.2, verbose=1, callbacks=[early_stop, PrintDot()])

    y_pred = model.predict(X_test).flatten()

    print('\n')
    print('exp_var', metrics.explained_variance_score(y_test, y_pred))
    print('mean_ae', metrics.mean_absolute_error(y_test, y_pred))
    print('mrse', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('median_ae', metrics.median_absolute_error(y_test, y_pred))
    print('\n')

predictions_pipeline()
