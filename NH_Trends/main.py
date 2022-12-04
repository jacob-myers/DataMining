# Data Source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
# Author: Jake Myers

"""
Testing 2015:
Average score with 15 important features: 0.2816386355546581
Average score with full dataset: 0.2903930567306121
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Does a value exist in a column?: https://www.statology.org/pandas-check-if-value-in-column/
# Check null stuff: https://datatofish.com/check-nan-pandas-dataframe/


# Reads an IRS .xlsx file for a year in NH.
# Returns a dataframe, cleaned up data by ZIP.
def import_and_clean_irs(filename: str) -> pd.DataFrame:
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
    df.rename(columns={'Number of dependent exemptions': 'Number of dependents'}, inplace=True)  # 2017


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
def reshape_by_ZIP(df: pd.DataFrame) -> pd.DataFrame:

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


# Takes two dataframes, the year being looked at, and the previous year.
# Creates the change-in features, adds them to the current year df, and returns it.
def create_deltas(cur_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    cur_df['Change in returns since previous year'] = cur_df['Number of returns'] - prev_df['Number of returns']
    # Dependents count not present in 2019
    if 'Number of dependents' in cur_df.columns:
        cur_df['Change in dependents since previous year'] = cur_df['Number of dependents'] - prev_df['Number of dependents']
    return cur_df


# Takes a dataframe (of a year) and builds a RandomForestRegressor from it.
# returns the score and model.
def create_forest(df: pd.DataFrame) -> (float, object):
    # Target: Change in returns.
    # Features: Everything else.
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['Change in returns since previous year']),
        df['Change in returns since previous year'],
        test_size=.25,
        random_state=42)
    rf = RandomForestRegressor(max_depth=5)
    rf_reg = rf.fit(X_train, y_train)
    return rf.score(X_test, y_test), rf_reg


# Use PCA to get the n most important columns. Returns new dataframe.
# Reference: https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
def get_f_important_features(X, f: int, n_components: float):
    pca = PCA(n_components=n_components)
    pca_model = pca.fit_transform(X)

    # Number of components
    n_pcs = pca.components_.shape[0]
    # Index of most important feature in each component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    # Feature names
    initial_feature_names = [col for col in X.columns]
    # Putting feature names in order of importance
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # Uses the first f column names (most important) to build a new dataframe.
    return X[most_important_names[:f]]


"""
# Engineer features.
def engineer_tax_df(df: list[pd.DataFrame]) -> pd.DataFrame:
    pass
"""


# Runs all cleaning, reshaping, engineering, scoring on a year (needs prev year as well)
def test_year(year: int) -> None:
    raw_df = import_and_clean_irs(f"nh_{year}.xlsx")
    raw_prev_df = import_and_clean_irs(f"nh_{year - 1}.xlsx")

    df = reshape_by_ZIP(raw_df)
    df_prev = reshape_by_ZIP(raw_prev_df)

    df = create_deltas(df, df_prev)

    #for i in range(10):
    #    rf_score, rf_reg = create_forest(df)
    #    print(rf_score)

    avg_score = 0
    for i in range(100):
        rf_score, rf_reg = create_forest(df)
        avg_score += rf_score
    avg_score /= 100
    print(f'Average score with full dataset: {avg_score}')


if __name__ == '__main__':
    raw_2014 = import_and_clean_irs("nh_2014.xlsx")
    raw_2015 = import_and_clean_irs("nh_2015.xlsx")
    #raw_2016 = import_and_clean_irs("nh_2016.xlsx")
    #raw_2017 = import_and_clean_irs("nh_2017.xlsx")
    #raw_2018 = import_and_clean_irs("nh_2018.xlsx")
    #raw_2019 = import_and_clean_irs("nh_2019.xlsx")

    df_2014 = reshape_by_ZIP(raw_2014)
    df_2015 = reshape_by_ZIP(raw_2015)
    #df_2016 = reshape_by_ZIP(raw_2016)
    #df_2017 = reshape_by_ZIP(raw_2017)
    #df_2018 = reshape_by_ZIP(raw_2018)
    #df_2019 = reshape_by_ZIP(raw_2019)

    # Income tax before credit (2019) = Income tax (2015)
    # Note: cols_2015 = np.array(raw_2015.columns)

    #print(df_2015.info(verbose=True))

    #test_year(2019)


    """
    X_train, X_test, y_train, y_test = train_test_split(
        df_2015.drop(columns=['Change in returns since previous year']),
        df_2015['Change in returns since previous year'],
        test_size=.25,
        random_state=42)
    """

    df_2015 = create_deltas(df_2015, df_2014)
    important = get_f_important_features(df_2015.drop(columns=['Change in returns since previous year']), 15, 15)
    important['Change in returns since previous year'] = df_2015['Change in returns since previous year']
    #print(important.sample(10))

    avg_score_small_set = 0
    for i in range(100):
        rf_score, rf_reg = create_forest(important)
        avg_score_small_set += rf_score
    avg_score_small_set /= 100
    print(f'Average score with 15 important features: {avg_score_small_set}')

    test_year(2015)

    """
    pca = PCA(n_components=10)
    X = df_2015.drop(columns=['Change in returns since previous year'])
    pca_model = pca.fit_transform(X)
    n_pcs = pca.components_.shape[0]
    print(n_pcs)
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = [col for col in X.columns]
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    for i in most_important_names:
        print(i)
    """

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
