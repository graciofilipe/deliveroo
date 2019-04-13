# Data Science Exercise - Prep-Time

## Problem Definition
Deliveroo is committed to providing a delivery experience that delights our customers while still being incredibly efficient. Thus it is critical that we have the best possible model of how long it takes for a food order to be prepared. This allows us to ensure that a rider arrives at a restaurant to pick up an order exactly when the food is ready.
The aim of this exercise is to use historical data to predict the food preparation time for each order.

## Modelling Exercise
1. Use any tool or language you would prefer.
2. Perform any cleaning, exploration and/or visualisation on the provided data (orders.csv and restaurants.csv)
3. Build a model that estimates the time it takes for a restaurant to prepare the food for an order.
4. Evaluate the performance of this model and describe steps taken to improve the performance.

We recommend spending a maximum of 2 hours on this task. Feel free to include a list of additional ideas you would explore if you had more time to work on this problem. You will have the opportunity at an onsite interview to explain your approach and answer further questions on it.

Please return your code and explanation of your approach (separate report or Notebook) Please include any required instructions for how to run your code.


## Appendix: Data Schema and Description

Filename : **orders.csv**

| Column Name       | Type    | Description                   |
|-------------------|---------|-------------------------------|
| order_value       | Float   | Value of the order in local currency|
| order_created_at  | String (timestamp) | Timestamp (UTC) indicating when the customer places the order |
| restaurant_id     | Integer | Unique restaurant identifier |
| number_of_items   | Integer | Number of items in the order |
| order_acknowledged_at | String (timestamp) | Timestamp (UTC) indicating when the order is acknowledged by the restaurant. |
| order_ready_at    | String (timestamp) | Timestamp (UTC) indicating when the food is ready |

Filename : **restaurants.csv**

| Column Name       | Type    | Description                   |
|-------------------|---------|-------------------------------|
| restaurant_id     | Integer | Unique restaurant identifier  |
| country           | String  | Country where the restaurant is |
| city              | String  | City where the restaurant is |
| timezone_name     | String  | Name of the timezone         |
| type_of_food      | String  | Type of food prepared by the restaurant |
