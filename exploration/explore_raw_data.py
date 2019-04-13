import pandas as pd
import plotly as py
import plotly.graph_objs as go

from collections import Counter


### EXPLORE ORDERS
orders = pd.read_csv('data/orders.csv')

print('variables in orders', orders.columns)
print('number of orders', orders.shape[0])

orders['order_acknowledged_at'] = pd.to_datetime(orders['order_acknowledged_at'])
orders['order_created_at'] = pd.to_datetime(orders['order_created_at'])
orders['order_ready_at'] = pd.to_datetime(orders['order_ready_at'])

orders['seconds_from_created_to_ready'] = (orders['order_ready_at'] - orders['order_created_at']).dt.total_seconds()
orders['seconds_from_acknowledged_to_ready'] = (orders['order_ready_at'] - orders['order_acknowledged_at']).dt.total_seconds()


## GENERATE SOME FEATURES

# hour of the day for the order (kitchens will have busier and less busy times)
orders['hour_order_acknowledged_at'] = orders['order_acknowledged_at'].dt.hour

# average price per item for the restaurant (a proxy for the type of restaurant)
orders['value_per_item'] = orders['order_value'] / orders['number_of_items']

## number of orders per restaurant
orders_grouped_count = orders.groupby(by='restaurant_id').count()['number_of_items'] #dummy var for aggregation

orders_grouped_median = orders.groupby(by='restaurant_id').median()[['order_value', 'number_of_items', 'value_per_item']]


def run_orders_plots():
    # Histogram of relevant variables
    data = [go.Histogram(x=orders['order_value'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/order_value_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['number_of_items'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/number_of_items_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['seconds_from_created_to_ready'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/seconds_from_created_to_ready_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['seconds_from_acknowledged_to_ready'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/seconds_from_acknowledged_to_ready_hist.html', auto_open=False)

    data = [go.Histogram(x=orders_grouped_count)]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/orders_per_restaurant_hist.html', auto_open=False)

    data = [go.Histogram(x=orders['hour_order_acknowledged_at'])]
    fig = go.Figure(data=data)
    py.offline.plot(fig, filename='plots/hour_order_acknowledged_at_hist.html', auto_open=False)

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


run_orders_plots()


#EXPLORE RESTAURANTS DATA
restaurants = pd.read_csv('data/restaurants.csv')
print('variables in restaurants', restaurants.columns)
print(Counter(restaurants['city']))
print(Counter(restaurants['country']))
print(Counter(restaurants['type_of_food']))










