import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


def data_preparation(data_path):
    """
    loads the data and turns it into tensorflow datasets
    :return: (data, train_ds, val_ds, test_ds) (pandas_df, tf.Dataset, tf.Dataset, tf.Dataset)
            The original dataset and train, validation and test tensorflow datasets
    """
    data = pd.read_csv(data_path)
    data.drop(labels='Unnamed: 0', axis='columns', inplace=True)

    train, test = train_test_split(data, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    print(len(train), 'train examples')
    print(len(test), 'test examples')

    feature_names = ['order_value', 'number_of_items', 'hour_order_acknowledged_at',
                     'country', 'city', 'type_of_food', 'day_of_order', 'restaurant_id',
                     'rest_median_order_value', 'rest_median_number_of_items',
                     'rest_median_value_per_item']

    outcome_name = 'minutes_from_acknowledged_to_ready'

    batch_size = 32
    train_ds = df_to_dataset(train, features=feature_names, outcome=outcome_name, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, features=feature_names, outcome=outcome_name, batch_size=batch_size)
    test_ds = df_to_dataset(test, features=feature_names, outcome=outcome_name, shuffle=False, batch_size=batch_size)

    return data, train_ds, val_ds, test_ds

# A utility function to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, features, outcome, shuffle=True, batch_size=128):
    dataframe = dataframe.copy()
    y = dataframe.pop(outcome)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe[features]), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def prepare_feature_layer(data):
    """
    given the dataset, prepares the input to the tensorflow model
    :param data: pandas dataset to be used in the modeling
    :return: a tensorflow Features Layer object to serve as model layer
    """

    feature_columns = []

    # numeric cols
    for header in ['rest_median_order_value', 'rest_median_number_of_items',
                   'rest_median_value_per_item', 'number_of_items', 'order_value']:
      feature_columns.append(feature_column.numeric_column(header))

    # indicator cols - country
    country = feature_column.categorical_column_with_vocabulary_list(
          'country', list(data['country'].unique()))
    country_one_hot = feature_column.indicator_column(country)
    feature_columns.append(country_one_hot)

    # indicator cols - day_of_order
    day_of_order = feature_column.categorical_column_with_vocabulary_list(
          'day_of_order', list(data['day_of_order'].unique()))
    day_of_order_one_hot = feature_column.indicator_column(day_of_order)
    feature_columns.append(day_of_order_one_hot)

    # indicator cols - hour of acknowledgement
    hour_order_acknowledged_at = feature_column.categorical_column_with_vocabulary_list(
          'hour_order_acknowledged_at', list(data['hour_order_acknowledged_at'].unique()))
    hour_order_acknowledged_at_one_hot = feature_column.indicator_column(hour_order_acknowledged_at)
    feature_columns.append(hour_order_acknowledged_at_one_hot)

    # city embedding
    city = feature_column.categorical_column_with_vocabulary_list(
          'city', list(data['city'].unique()))
    city_embedding = feature_column.embedding_column(city, dimension=4)
    feature_columns.append(city_embedding)

    # type_of_food embedding
    type_of_food = feature_column.categorical_column_with_vocabulary_list(
          'type_of_food', list(data['type_of_food'].unique()))
    type_of_food_embedding = feature_column.embedding_column(type_of_food, dimension=8)
    feature_columns.append(type_of_food_embedding)

    # restaurant_id embedding
    restaurant_id = feature_column.categorical_column_with_vocabulary_list(
          'restaurant_id', list(data['restaurant_id'].unique()))
    restaurant_id_embedding = feature_column.embedding_column(restaurant_id, dimension=22)
    feature_columns.append(restaurant_id_embedding)

    # crossed cols
    crossed_feature = feature_column.crossed_column([hour_order_acknowledged_at, day_of_order], hash_bucket_size=100)
    crossed_feature = feature_column.indicator_column(crossed_feature)
    feature_columns.append(crossed_feature)

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    return feature_layer


def model_building(feature_layer):
    """
    builds and returns tensorflow model
    :param feature_layer: the feature layer built from the dataset
    :return: tensforflow model
    """

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(optimizer=optimizer,
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def model_fit_and_evaluate(model, train_ds, val_ds, test_ds, epochs):
    """
    fits the model and presents the model error
    :param model: tensorflow model to train
    :param train_ds: tf.Dataset of training samples
    :param val_ds: tf.Dataset of validation samples
    :param test_ds: tf.Dataset of test samples
    :param epochs: the number of epochs to run trhough the data
    :return: None - print errors to standard out
    """

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=3)
    model.fit(train_ds,
              verbose=2,
              validation_data=val_ds,
              callbacks=[early_stop],
              epochs=epochs)

    los, mae, mse = model.evaluate(test_ds, verbose=1)
    print('test_loss', los)
    print('test mae', mae)
    print('test mse', mse)

    #todo: model saving



def modeling_pipeline():
    """
    runs the full modeling pipeline
    :return: None
    """
    data_path = './processed_data/order_and_restaurant_filtered_and_enriched.csv'
    data_df, train_ds, val_ds, test_ds = data_preparation(data_path)
    feature_layer = prepare_feature_layer(data_df)
    model = model_building(feature_layer)
    model_fit_and_evaluate(model=model,
                           train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
                           epochs = 111)

if __name__ == '__main__':
    modeling_pipeline()