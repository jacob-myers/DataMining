# Data Source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
# Author: Jake Myers

import pandas as pd
import numpy as np

# Does a value exist in a column?: https://www.statology.org/pandas-check-if-value-in-column/


def import_and_clean_irs(filename: str):
    # CLEAN COLUMNS
    # Reading in data sheet with a multi-level column index.
    # https://stackoverflow.com/questions/69855943/pandas-read-csv-with-columns-name-spread-over-multiple-rows
    df = pd.read_excel(filename, skiprows=3, header=[0, 1, 2])

    # Drop the garbage column index (3rd level of column index).
    # https://www.geeksforgeeks.org/how-to-drop-a-level-from-a-multi-level-column-index-in-pandas-dataframe/
    df.columns = df.columns.droplevel(2)

    # Merge level 1 and 2 of multi-level column index to make single level.
    # https://stackoverflow.com/questions/24290297/pandas-dataframe-with-multiindex-column-merge-levels
    df.columns = df.columns.map(' | '.join).str.strip(' | ')

    # Clean up the names (remove '[number]' and Unnamed columns (from multi-level step)
    # https://stackoverflow.com/questions/61653697/pandas-str-replace-with-regex
    df.columns = df.columns.str.replace(r' \| Unnamed.*', '', regex=True)
    df.columns = df.columns.str.replace(r' \[\d*\]', '', regex=True)
    df.columns = df.columns.str.replace(r'\n', '', regex=True)

    # CLEAN DATA
    # TO DO
    # Rename columns
    df.rename(columns={'ZIPcode':'ZIP'}, inplace=True)

    # Remove blank lines (Where ZIP code is blank) and end lines (dumb info)
    # Forcing it to numeric, and deleting all non numeric, so Na and NaN
    # https://stackoverflow.com/questions/42125131/delete-row-based-on-nulls-in-certain-columns-pandas
    df['ZIP'] = pd.to_numeric(df['ZIP'], errors='coerce')
    df.dropna(axis=0, subset=['ZIP'], inplace=True)

    # Catch all ZIP code, not real.
    # https://www.transportationinsight.com/resources/where-is-zip-code-99999-a-piece-of-clean-data-makes-a-big-difference/
    indices = df[df['ZIP'] == 99999.0].index
    df.drop(indices, inplace=True)
    # IRS stopped recording this ZIP after 2015. Too small maybe? ~100 returns in 2015.
    indices = df[df['ZIP'] == 03291.0].index
    df.drop(indices, inplace=True)

    # Reset index and delete old index.
    df.reset_index(inplace=True)
    del df['index']

    return df


if __name__ == '__main__':
    df_2015 = import_and_clean_irs("nh_2015.xlsx")
    #df_2016 = import_and_clean_irs("nh_2016.xlsx")
    #df_2017 = import_and_clean_irs("nh_2017.xlsx")
    #df_2018 = import_and_clean_irs("nh_2018.xlsx")
    #df_2019 = import_and_clean_irs("nh_2019.xlsx")

    #Income tax before credit (2019) = Income tax (2015)

    cols_2015 = np.array(df_2015.columns)

    # Same amount of empty rows?
    """
    print(df_2015.isnull().sum().sum())
    print(df_2016.isnull().sum().sum())
    print(df_2017.isnull().sum().sum())
    print(df_2018.isnull().sum().sum())
    print(df_2019.isnull().sum().sum())
    """

    # Identify ZIP codes that exist in 2015 and not 2016.
    # https://stackoverflow.com/questions/26518802/pandas-iterating-through-each-row
    """
    rows = []
    for row in df_2016.iterrows():
        if pd.isna(row[1]['Size of adjusted gross income']):
            rows.append(row[1]['ZIP'])

    for row in df_2015.iterrows():
        if pd.isna(row[1]['Size of adjusted gross income']):
            if row[1]['ZIP'] not in rows:
                print(row[1]['ZIP'])
    """

    # Columns that exist every year.
    """
    for col in df_2015.columns:
        if col in df_2019.columns and col in df_2018.columns and col in df_2017 and col in df_2016:
            print(col)
    """
