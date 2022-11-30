# Data Source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
# Author: Jake Myers

import pandas as pd
import numpy as np

# Does a value exist in a column?: https://www.statology.org/pandas-check-if-value-in-column/
# Check null stuff: https://datatofish.com/check-nan-pandas-dataframe/


# Reads an IRS .xlsx file for a year in NH.
# Returns a dataframe, cleaned up data by ZIP.
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

    # Drop year totals (can be calculated, also not accurate after other drops).
    indices = df[df['ZIP'] == 00000.0].index
    df.drop(indices, inplace=True)

    # Reset index and delete old index.
    df.reset_index(inplace=True)
    del df['index']

    return df


# Takes a dataframe, the tax data by zip/size of gross income.
# Returns a dataframe, the tax data by zip, with each size as a feature.
def reshape_by_ZIP(df: pd.DataFrame):
    
    # Dataframe with just totals for each zip.
    df_totals = df.loc[df['Size of adjusted gross income'].isnull()]
    df_totals = df_totals.set_index('ZIP')
    del df_totals['Size of adjusted gross income']

    # Dataframe with all rows of specific brackets.
    df_per_bracket = df.loc[df['Size of adjusted gross income'].notnull()]

    """
    # Ensure df_new and df_old contain everything from df
    print(df.info())
    print(df_totals.info())
    print(df_per_bracket.info())
    """

    # Pivot data: https://stackoverflow.com/questions/44725441/create-new-columns-based-on-unique-values-of-values-in-pandas
    # Pick just three columns (Amount of returns in each bracket).
    df_per_bracket = df_per_bracket[['ZIP', 'Size of adjusted gross income', 'Number of returns']]
    # New dataframe generated: Index is the zip, columns are the brackets, values are the amount of returns.
    df_pivoted = df_per_bracket.pivot_table(values='Number of returns', columns='Size of adjusted gross income', index='ZIP')

    # Renaming columns in pivoted dataframe
    df_pivoted.rename(columns={
        '$1 under $25,000': 'Number of returns with income between $1 and $25,000',
        '$25,000 under $50,000': 'Number of returns with income between $25,000 and $50,000',
        '$50,000 under $75,000': 'Number of returns with income between $50,000 and $75,000',
        '$75,000 under $100,000': 'Number of returns with income between $75,000 and $100,000',
        '$100,000 under $200,000': 'Number of returns with income between $100,000 and $200,000',
        '$200,000 or more': 'Number of returns with income $200,000 or more',
        }, inplace=True)

    # Merge the totals with the pivoted df. (Adding amount of returns in a bracket as a column, 6 total)
    df_combined = df_totals.merge(df_pivoted, how='inner', on=['ZIP'])

    # Rearrange the columns.
    # Number of returns, then cols for returns in each bracket, then the rest of the cols.
    df_combined_cols = np.array(df_combined.columns).tolist()
    df_combined_cols = np.array(df_combined.columns).tolist()
    first_cols = ['Number of returns',
                 'Number of returns with income between $1 and $25,000',
                 'Number of returns with income between $25,000 and $50,000',
                 'Number of returns with income between $50,000 and $75,000',
                 'Number of returns with income between $75,000 and $100,000',
                 'Number of returns with income between $100,000 and $200,000',
                 'Number of returns with income $200,000 or more']
    for col in first_cols:
        if col in df_combined_cols:
            df_combined_cols.remove(col)

    df_combined = df_combined[first_cols + df_combined_cols]

    # Reset index (currently ZIP, so that ZIP is a column again)
    df_combined = df_combined.reset_index()
    return df_combined


if __name__ == '__main__':
    raw_2015 = import_and_clean_irs("nh_2015.xlsx")
    #raw_2016 = import_and_clean_irs("nh_2016.xlsx")
    #raw_2017 = import_and_clean_irs("nh_2017.xlsx")
    #raw_2018 = import_and_clean_irs("nh_2018.xlsx")
    #raw_2019 = import_and_clean_irs("nh_2019.xlsx")

    df_2015 = reshape_by_ZIP(raw_2015)
    #df_2016 = reshape_by_ZIP(raw_2016)
    #df_2017 = reshape_by_ZIP(raw_2017)
    #df_2018 = reshape_by_ZIP(raw_2018)
    #df_2019 = reshape_by_ZIP(raw_2019)

    # Income tax before credit (2019) = Income tax (2015)
    # Note: cols_2015 = np.array(raw_2015.columns)

    print(df_2015.info(verbose=True))



    """
    # All are 229, meaning one null (Size of adjusted gross income) per ZIP
    # This is the total for the year. Meaning there are no other nulls in the DB
    print(df_2015.isnull().sum().sum())
    print(df_2016.isnull().sum().sum())
    print(df_2017.isnull().sum().sum())
    print(df_2018.isnull().sum().sum())
    print(df_2019.isnull().sum().sum())

    print(df_2015['ZIP'].nunique())
    print(df_2015['ZIP'].nunique())
    print(df_2015['ZIP'].nunique())
    print(df_2015['ZIP'].nunique())
    print(df_2015['ZIP'].nunique())
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
