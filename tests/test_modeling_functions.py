from model.fit_and_predict_with_tf2 import data_preparation, prepare_feature_layer, model_building
import pandas as pd

def test_data_preparation():
    '''
    test the right variables are built and returned
    :return:
    '''
    data_path = './fixtures/model_data_sample.csv'
    data, train_ds, val_ds, test_ds = data_preparation(data_path)

    assert type(data) == type(pd.DataFrame())
    assert str(type(train_ds)) == "<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>"
    assert str(type(val_ds)) == "<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>"
    assert str(type(test_ds)) == "<class 'tensorflow.python.data.ops.dataset_ops.BatchDataset'>"


def test_prepare_feature_layer():
    '''
    test the right variables are built and returned
    :return:
    '''
    data_path = './fixtures/model_data_sample.csv'
    data, train_ds, val_ds, test_ds = data_preparation(data_path)
    feature_layer = prepare_feature_layer(data)
    assert str(type(feature_layer)) == "<class 'tensorflow.python.feature_column.feature_column_v2.DenseFeatures'>"


def test_model_building():
    '''
    test it builds and returns model
    :return:
    '''
    data_path = './fixtures/model_data_sample.csv'
    data, train_ds, val_ds, test_ds = data_preparation(data_path)
    feature_layer = prepare_feature_layer(data)
    model = model_building(feature_layer)
    assert str(type(model)) == "<class 'tensorflow.python.keras.engine.sequential.Sequential'>"
