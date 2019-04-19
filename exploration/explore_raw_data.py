import pandas as pd
import plotly as py
import plotly.graph_objs as go

from collections import Counter


"""
This script explores the dataset and produces histogram plots of the main variables
It also generates some new features based on the timestamp data and explore them
"""

def explore_orders(orders):
    """
    generates histograms from existing and constructed features from the 'orders' dataset
    :param orders: pandas dataframe of the orders dataset
    :return: None (write html files)
    """

    print('variables in orders', orders.columns)
    print('\n')
    print('number of orders', orders.shape[0])
    print('\n')

    # convert to timestapms and generating some new variables
    orders['order_acknowledged_at'] = pd.to_datetime(orders['order_acknowledged_at'])
    orders['order_created_at'] = pd.to_datetime(orders['order_created_at'])
    orders['order_ready_at'] = pd.to_datetime(orders['order_ready_at'])

    # food preparation time
    orders['minutes_from_acknowledged_to_ready'] = (orders['order_ready_at'] - orders['order_acknowledged_at']).dt.total_seconds() /60

    print('orders ready in 0 minutes:', sum(orders['minutes_from_acknowledged_to_ready'] <= 0))

    # hour of the day for the order (kitchens will have busier and less busy times)
    orders['hour_order_acknowledged_at'] = orders['order_acknowledged_at'].dt.hour

    # day of the week for the order (kitchens will have busier and less busy times)
    orders['day_of_order'] = orders['order_acknowledged_at'].dt.dayofweek #  Monday=0, Sunday=6.

    # average price per item for the restaurant (a proxy for the type of restaurant)
    orders['value_per_item'] = orders['order_value'] / orders['number_of_items']

    ## number of orders per restaurant
    orders_grouped_count = orders.groupby(by='restaurant_id').count()['number_of_items'] #dummy var for aggregation

    orders_grouped_median = orders.groupby(by='restaurant_id').median()[['order_value', 'number_of_items', 'value_per_item']]

    # Histogram of relevant variables
    data = [go.Histogram(x=orders['order_value'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/order_value_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['number_of_items'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/number_of_items_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['minutes_from_acknowledged_to_ready'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/minutes_from_acknowledged_to_ready_hist.html', auto_open=False)

    data = [go.Histogram(x=orders_grouped_count)]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/orders_per_restaurant_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['hour_order_acknowledged_at'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/hour_order_acknowledged_at_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['day_of_order'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/day_of_order_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['value_per_item'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/value_per_item_hist.html', auto_open=False)

    data = [go.Histogram(x=orders_grouped_median['value_per_item'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/median_value_per_item_over_restaurant_hist.html', auto_open=False)

    data = [go.Histogram(x=orders_grouped_median['number_of_items'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/median_number_of_items_over_restaurant_hist.html', auto_open=False)

    data = [go.Histogram(x=orders_grouped_median['order_value'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/median_order_value_over_restaurant_hist.html', auto_open=False)


def explore_restaurants(restaurants):
    """
    summarises the main variables in the restaurants dataset
    :param restaurants: pandas dataframe of the restaurants data set
    :return: None (print to standard out)
    """

    print('number of restaurants', restaurants.shape[0])
    print('\n')

    #EXPLORE RESTAURANTS DATA
    print('variables in restaurants', restaurants.columns)
    print('\n')

    print(Counter(restaurants['city']))
    print('\n')

    print(Counter(restaurants['country']))
    print('\n')

    print(Counter(restaurants['type_of_food']))
    print('\n')


def merge_orders_and_restaurants_and_explore(orders, restaurants):
    """
    :param orders: the orders dataset in a pandas dataframe
    :param restaurants: the restaurants dataset in a datafame
    :return: None (prints info to standard out)
    """

    order_and_rest = orders.merge(right=restaurants, how='left', on='restaurant_id')

    print('restaurant info at the order level')
    print('\n')
    print(Counter(order_and_rest['city']))
    print('\n')
    print(Counter(order_and_rest['country']))
    print('\n')
    print(Counter(order_and_rest['type_of_food']))
    print('\n')


def main_explore():
    """
    execute the exploration pipeline
    :return:
    """
    restaurants = pd.read_csv('raw_data/restaurants.csv')
    orders = pd.read_csv('raw_data/orders.csv')

    explore_orders(orders=orders)
    explore_restaurants(restaurants=restaurants)
    merge_orders_and_restaurants_and_explore(orders, restaurants)


if __name__ == '__main__':
    main_explore()






