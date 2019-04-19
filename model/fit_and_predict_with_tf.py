import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression

from sklearn.decomposition import TruncatedSVD

from scipy.stats import randint as sp_randint


import plotly as py
import plotly.graph_objs as go

import numpy as np
np.random.seed(42)


def model_data_preparation(feature_names, outcome_name):
    data = pd.read_csv('processed_data/order_and_restaurant_filtered_and_enriched.csv')
    data.drop(labels='Unnamed: 0', axis='columns', inplace=True)

    X = data[feature_names]
    y = data[outcome_name]

    return X, y

def build_modelling_pipeline(n_iter=2):

    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_features = ['order_value', 'number_of_items',
                        'hour_order_acknowledged_at',
                        'rest_median_order_value',
                        'rest_median_number_of_items',
                        'rest_median_value_per_item']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['country', 'city', 'type_of_food', 'day_of_order']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])




    # MAIN MODEL

    # for feature reduction
    svd = TruncatedSVD()

    forest_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('svd', svd),
                                  ('regressor', RandomForestRegressor())])

    # specify parameters and distributions to sample from
    param_dist = {"regressor__n_estimators":sp_randint(30, 111),
                  "regressor__max_depth": sp_randint(7, 20),
                  "regressor__max_features": sp_randint(10, 30),
                  "svd__n_components": sp_randint(30, 40)}


    # run randomized search
    n_iter_search = n_iter
    random_forest_search = RandomizedSearchCV(forest_pipe,verbose=3,
                                              param_distributions=param_dist,
                                              n_iter=n_iter_search, cv=3)


    # LINEAR BENCHMARK (not even hyper parameter search)
    linear_benchmark = Pipeline(steps=[('preprocessor', preprocessor),
                          ('regressor', LinearRegression())])

    model_dict = {'main_model':random_forest_search,
                  'benchmark_model': linear_benchmark}
    return model_dict



def train_models(X, y, model_dict):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()


    for model_name, model in model_dict.items():

        print('training', model_name)
        print('\n')

        model.fit(X_train, y_train)
        y_test_model = model.predict(X_test)

        print('model', model_name, 'metrics')
        print('exp_var', metrics.explained_variance_score(y_test, y_test_model))
        print('mean_ae',metrics.mean_absolute_error(y_test, y_test_model))
        print('mrse',np.sqrt(metrics.mean_squared_error(y_test, y_test_model)))
        print('median_ae',metrics.median_absolute_error(y_test, y_test_model))
        print('\n')

        trace = go.Scatter(x=y_test, y=y_test_model, mode='markers')
        layout = go.Layout(xaxis=dict(title='real_data'),
                           yaxis=dict(title='predicted'),
                           title=model_name)
        fig = go.Figure(data=[trace], layout=layout)
        py.offline.plot(fig, filename='plots/{}_fit.html'.format(model_name), auto_open=False)

def predictions_pipeline(n_iter):

    feature_names = ['order_value', 'number_of_items', 'hour_order_acknowledged_at',
                     'country', 'city', 'type_of_food', 'day_of_order',
                     'rest_median_order_value', 'rest_median_number_of_items',
                     'rest_median_value_per_item']

    outcome_name = ['minutes_from_acknowledged_to_ready']

    X, y = model_data_preparation(feature_names, outcome_name)
    model_dict = build_modelling_pipeline(n_iter=n_iter)
    train_models(X, y, model_dict)

if __name__ == '__main__':
    predictions_pipeline(n_iter=15)