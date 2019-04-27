import pandas as pd
from collections import Counter

"""
execute the ETL stage of the pipeline. taking in the original datasets, calculating new features
cleaning and returing a final dataset for the prediction model in the next stage
"""

def order_feature_generation(orders):
    """
    generates new features for the orders dataset and returns it
    :param orders: pandas dataframe of the orders dataset
    :return: pandas dataframe with more features including the outcome to model
    """

    orders['order_acknowledged_at'] = pd.to_datetime(orders['order_acknowledged_at'])
    orders['order_created_at'] = pd.to_datetime(orders['order_created_at'])
    orders['order_ready_at'] = pd.to_datetime(orders['order_ready_at'])
    orders['minutes_from_acknowledged_to_ready'] = (orders['order_ready_at'] - orders['order_acknowledged_at']).dt.total_seconds() / 60

    # hour of the day for the order (kitchens will have busier and less busy times)
    orders['hour_order_acknowledged_at'] = orders['order_acknowledged_at'].dt.hour

    # day of the week for the order (kitchens will have busier and less busy times)
    orders['day_of_order'] = orders['order_acknowledged_at'].dt.dayofweek #  Monday=0, Sunday=6.

    # average price per item for the restaurant (a proxy for the type of restaurant)
    orders['value_per_item'] = orders['order_value'] / orders['number_of_items']

    return orders


def restaurant_feature_generation_from_orders(orders, restaurants):
    """
    generate features at the restaurant level
    :param orders: pandas dataframe of the orders dataset
    :param restaurants: pandas dataframe with the restaurants dataset
    :return: pandas datafarme consisting of the restaurants with new features
    """

    # some medians at the restaurant level
    orders_grouped_median = orders.groupby(by='restaurant_id').median()[['order_value', 'number_of_items', 'value_per_item']]

    orders_grouped_median.rename(mapper={'order_value': 'rest_median_order_value',
                                         'number_of_items': 'rest_median_number_of_items',
                                         'value_per_item': 'rest_median_value_per_item'},
                                 axis ='columns',
                                 inplace=True)

    orders_grouped_median.reset_index(level=0, inplace=True)

    restaurants_enriched = restaurants.merge(orders_grouped_median, how='inner', on='restaurant_id')

    print('shape of original restaurant data', restaurants.shape[0])
    print('shape or enriched restaurant data', restaurants_enriched.shape[0])

    return restaurants_enriched


def filter_orders_data(orders, max_items=25, max_value=200, min_minutes=0.1, max_minutes=90):
    """
    removes some outliers in the orders data
    :param orders: pandas dataframe of orders data
    :return: pandas dataframe of filtered orders data
    """

    # orders with more than 25 items (confer number_of_items_hist.html)
    orders_filtered = orders.loc[orders['number_of_items'] <= max_items]

    # orders that total more than 200 in local currency (GBP/EUR) (confer order_value_hist.html)
    orders_filtered = orders_filtered.loc[orders_filtered['order_value'] <= max_value]

    orders_filtered = orders_filtered.loc[orders_filtered['minutes_from_acknowledged_to_ready'] > min_minutes]
    orders_filtered = orders_filtered.loc[orders_filtered['minutes_from_acknowledged_to_ready'] < max_minutes]

    print('size of orders after removals', orders_filtered.shape[0])

    return orders_filtered

def merge_and_filter_on_location_and_food_type(orders, restaurants):
    """
    merges the orders and restaurants datasets, and removes outliers for the model
    :param orders: pandas dataframe of orders
    :param restaurants: pandas dataframe of restaurants
    :return: pandas dataframe of orders and restaurants after filtering
    """

    order_and_rest = orders.merge(right=restaurants, how='inner', on='restaurant_id')
    assert order_and_rest.shape[0] == orders.shape[0]

    # keep only cities and food types with more than 100 orders
    city_counter = Counter(order_and_rest['city'])
    cities_to_keep = [c for c, n in city_counter.items() if n > 100]

    food_type_counter = Counter(order_and_rest['type_of_food'])
    food_type_to_keep = [c for c, n in food_type_counter.items() if n > 100]

    order_and_rest_filtered = order_and_rest.loc[order_and_rest['city'].isin(cities_to_keep)]
    order_and_rest_filtered = order_and_rest_filtered.loc[order_and_rest_filtered['type_of_food'].isin(food_type_to_keep)]

    print('final size of dataset after mergeing and filtering', order_and_rest_filtered.shape)

    return order_and_rest_filtered


def etl_main():
    """
    runs the etl pipeline
    :return: None (writes final dataset to csv)
    """
    orders = pd.read_csv('raw_data/orders.csv')
    restaurants = pd.read_csv('raw_data/restaurants.csv')

    orders.drop(labels='Unnamed: 0', axis='columns', inplace=True)
    orders.dropna(inplace=True)
    restaurants.drop(labels='Unnamed: 0', axis='columns', inplace=True)
    restaurants.dropna(inplace=True)

    print('orders original size', orders.shape[0])

    orders_enriched = order_feature_generation(orders)

    orders_enriched_filtered = filter_orders_data(orders_enriched,
                                                  max_items=25, max_value=200,
                                                  min_minutes=0.1, max_minutes=90)

    restaurants_enriched = restaurant_feature_generation_from_orders(orders_enriched_filtered,
                                                                     restaurants)

    final_data = merge_and_filter_on_location_and_food_type(orders=orders_enriched_filtered,
                                                            restaurants=restaurants_enriched)

    print('final data size', final_data.shape)

    final_data.to_csv('processed_data/order_and_restaurant_filtered_and_enriched.csv')

if __name__ == '__main__':
    etl_main()