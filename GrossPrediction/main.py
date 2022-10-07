"""Program for identifying and isolating relevant data for gross predictions.

    Author: Jake Myers
    Class: DAT-330
    Assignment: Data Cleaning and Packages

Certification of Authenticity:
I certify that this is entirely my own work, except where I have given
fully-documented references to the work of others. I understand the definition
and consequences of plagiarism and acknowledge that the assessor of this
assignment may, for the purpose of assessing this assignment:
- Reproduce this assignment and provide a copy to another member of academic
- staff; and/or Communicate a copy of this assignment to a plagiarism checking
- service (which may then retain a copy of this assignment on its database for
- the purpose of future plagiarism checking)
"""

from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
import numpy as np


def get_numerical_columns(df: pd.DataFrame) -> list[str]:
    """Find all column names that contain numerical data.

    :param df: pd.DataFrame to get columns from.
    """
    numerical_cols: list[str] = []
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            numerical_cols.append(col)
    return numerical_cols


def order_relevance(df: pd.DataFrame, y_col: str, x_possibles: list[str], x_currents: list[str] = []) -> list[str]:
    """Find which numerical columns (x) are most relevant to modelling a column (y).

    Starts by identifying the single most relevant, passes it recursively,
    finding the next most relevant (highest score when combined), and so
    on until all columns have been tested and appended.

    :param df: pd.DataFrame to be explored.
    :param y_col: String, the value that relevance is being tested against.
    :param x_possibles: list of strings, the names of numeric columns left un-assigned.
    :param x_currents: list of strings, ordered list of relevant columns.

    :return The list built on x_currents. An ordered list starting with the most relevant.
    """
    gross_predictor = linear_model.LinearRegression()

    # Base case
    if len(x_possibles) == 1:
        return x_currents + x_possibles

    # Recursive case
    cur_max_col = ""
    cur_max_score = 0
    # Finds the highest score possible with x_possibles
    for col in x_possibles:
        x_temp = x_currents.copy()
        x_temp.append(col)
        x = np.array(df[x_temp])
        y = np.array(df[y_col]).reshape(-1,1)
        score = gross_predictor.fit(x,y).score(x,y)
        if score > cur_max_score:
            cur_max_score = score
            cur_max_col = col
    x_possibles.remove(cur_max_col)
    x_currents.append(cur_max_col)
    return order_relevance(df, y_col, x_possibles, x_currents)


if __name__ == '__main__':
    # Read csv into pd.DataFrames.
    df_pop = pd.read_csv("HS-STAT-Population-of-Vermont-towns-1930-2019.csv", skiprows=4)
    df_tax = pd.read_csv("vt_sales_and_use.csv")

    # Using Melt to format population data.
    df_pop.drop('CTC', axis=1, inplace=True)  # Axis 0 = row, Axis 1 = col. dropping CTC column
    df_pop = df_pop.melt(id_vars=['NAME'])
    df_pop.rename(columns={'NAME': 'town', 'variable': 'year', 'value': 'population'}, inplace=True)
    df_pop = df_pop.astype({'year':'int64'})

    # Merging population data into tax data.
    df_combined = df_tax.merge(df_pop, how='inner', on=['year', 'town'])

    # Aggregating data (removing month parts and summing their values)
    df_agg = df_combined.groupby(by=["town", "year", "type", "population"], as_index=False).sum(numeric_only=True)
    # Irrelevant numerical data (sum makes it nonsense)
    df_agg.drop('month_num', axis=1, inplace=True)

    # Use functions to identify the order of relevance of numerical data.
    numerical_cols = get_numerical_columns(df_agg)
    numerical_cols.remove('gross')
    order_of_relevance = order_relevance(df_agg, 'gross', numerical_cols)
    print(order_of_relevance)

    # Score using the 3 most relevant numerical columns.
    gross_predictor = linear_model.LinearRegression()
    x = np.array(df_agg[order_of_relevance[0:3]])
    y = np.array(df_agg['gross']).reshape(-1, 1)
    print(gross_predictor.fit(x, y).score(x, y))

    """
    # Redefining towns by mapping each town to an integer
    ordinal_encoder = OrdinalEncoder()
    # Turns series (town column) into a numpy array (column vector) and reshape turns it into a row vector.
    town_data_reshaped = np.array(df_combined['town']).reshape(-1, 1)
    # Fit transform maps the towns to integers. Needs a row vector.
    town_data_encoded = ordinal_encoder.fit_transform(town_data_reshaped)
    """

    # Using One Hot Encoder to redefine data structure
    town_oh_encoder = OneHotEncoder(sparse=False)  # sparse=False prevents it from creating a sparse matrix. just a regular one
    # Turns series (town column) into a numpy array (column vector) and reshape turns it into a row vector.
    town_data_reshaped = np.array(df_combined['town']).reshape(-1, 1)
    # Creates essentially a reduced row echelon matrix with town names as column headers.
    town_data_1hot = town_oh_encoder.fit_transform(town_data_reshaped)  # Default stores as a sparse array (very little memory)
    # town_data_1hot.toarray()

    # Concat one hot representation onto combined dataframe.
    data_final = pd.DataFrame(town_data_1hot, columns=df_combined['town'].unique())  # A column for each town
    data_concat = pd.concat([df_combined, data_final], axis=1)

    # Use functions to identify the order of relevance of numerical data.
    numerical_cols = get_numerical_columns(data_concat)
    numerical_cols.remove('gross')
    order_of_relevance = order_relevance(data_concat, 'gross', numerical_cols)
    print(order_of_relevance)
    print(numerical_cols)
