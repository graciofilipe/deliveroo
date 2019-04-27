from cleaning_and_etl.etl import filter_orders_data
import pandas as pd


def test_filter_orders_data():

    orders_df = pd.DataFrame(data={
        'number_of_items':[10, 12, 5, 2, 3, 5, 60],
        'order_value': [10, 34, 140, 54, 3, 49, 54],
        'minutes_from_acknowledged_to_ready': [4, 0.1, 64, 43, 32, 13, 43]
    })
    filtered = filter_orders_data(orders_df, max_items=10, max_value=100, min_minutes=1, max_minutes=50)
    assert filtered.shape == (4, 3)