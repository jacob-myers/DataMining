# Data Source: https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-zip-code-data-soi
# Author: Jake Myers

"""
Testing 2015 (No Engineering, original parameters for regressor):
Average score with 15 important features: 0.2816386355546581
Average score with full dataset: 0.2903930567306121

Testing 2015 (No engineering, adjusted parameters)
Average score: 0.35989386105395005

Final Results:
(Default)
Average Score 2015: 0.3835141409442009
Average Score 2016: -0.3358394993762476
Average Score 2017: 0.6330135886859775
Average Score 2018: 0.3984971955626156
Average Score 2019: 0.3778606594504725
Average score with years combined: 0.23213134427052276

(After Full Normalization Method)
Average Score 2015: 0.867783280215804
Average Score 2016: 0.6876975453932882
Average Score 2017: 0.8039867999897643
Average Score 2018: 0.8021215802992412
Average Score 2019: 0.7468203920532035
Average score with years combined: 0.8598821856103147

(Examining fully normalized data)
Most important features (from custom PCA method):
2015: ['Total itemized deductions | Amount of AGI', 'Salaries and wages in AGI | Amount', 'ZIP', 'Taxable income | Amount', 'Pensions and annuities in AGI | Amount']
2016: ['Total itemized deductions | Amount of AGI', 'Salaries and wages in AGI | Amount', 'ZIP', 'Pensions and annuities in AGI | Amount', 'Net capital gain (less loss) in AGI | Amount']
2017: ['Total itemized deductions | Amount of AGI', 'Salaries and wages in AGI | Amount', 'ZIP', 'Taxable income | Amount', 'Taxable income | Amount']
2018: ['Total itemized deductions | Amount of AGI', 'Salaries and wages in AGI | Amount', 'ZIP', 'Total itemized deductions | Amount of AGI', 'Qualified business income deduction | Amount']
2019: ['Total itemized deductions | Amount of AGI', 'Salaries and wages in AGI | Amount', 'ZIP', 'Total itemized deductions | Amount of AGI', 'Taxable income | Amount']
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import zscore
from scipy import stats

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
    # criterion = 'absolute_error' has shown the most success. Much slower run time.
    # Max depth seems best around 5-7
    # Min samples split doesn't change too much, maybe best around 7
    rf = RandomForestRegressor(criterion="absolute_error", max_depth=6, min_samples_split=7)
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


# Use df.var() to find the features that explain the most variance in the dataset.
def get_f_variant_features(df: pd.DataFrame, f: int):
    variant_features = []
    var_df = df.var()
    for feat in var_df.nlargest(f):
        variant_features.append(var_df[var_df == feat].index[0])
    return variant_features


# Not used.
def turn_into_pca_components(X, n_components: float):
    pca = PCA(n_components=n_components)
    pca_model = pca.fit_transform(X)
    print(pd.DataFrame(pca.components_).info())
    return pd.DataFrame(pca.components_)


# Normalize n most variant features.
# Has no noticeable effect on score
def normalize_features(df: pd.DataFrame, n: int) -> pd.DataFrame:
    var_df = df.var()
    df_new = df.copy(deep=True)

    # Normalize n features with the greatest variance.
    for feat in var_df.nlargest(n):
        #print(var_df[var_df == feat].index[0]) # Index of feat (index for series is column name).
        col_name = var_df[var_df == feat].index[0]
        df_new[col_name] = zscore(df_new[col_name])
        #print(df[col_name])

    """
    for col in df.columns:
        df[col] = zscore(df[col])
    """

    return df_new


# Returns a dataframe with all dataframes appended. Drops columns with NaNs.
# When modeled on, provides a very poor score.
def combine_all_years(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    df_total = dfs[0]
    dfs.pop(0)
    for df in dfs:
        df_total = df_total.append(df, ignore_index=True)
    df_total = df_total.dropna(axis=1, how='any') # If there are any NaN's in a col, drop it.
    return df_total


# DEPRECATED, use test_dataframe
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


# Tests creation of a forest on given dataframe 'runs' number of times
def test_dataframe(df: pd.DataFrame, runs: int) -> None:
    avg_score = 0
    for i in range(runs):
        # Does train test split inside create_forest.
        rf_score, rf_reg = create_forest(df)
        avg_score += rf_score
        print(rf_score)
    avg_score /= runs
    print(f'Average score: {avg_score}')


# Not used.
# Show variance by different methods.
def test_variance(df: pd.DataFrame) -> None:
    # Most variant features
    features = get_f_variant_features(df, 5)
    print(features)

    # Normalize some pieces and then see most variant features
    engineered_df = normalize_features(df, 25)
    features_eng = get_f_variant_features(engineered_df, 5)
    print(features_eng)

    # Look at most important features according to PCA
    features_imp = get_f_important_features(df.drop(columns=['Change in returns since previous year']), 5, 5)
    print(list(features_imp))


if __name__ == '__main__':
    # Import data and clean it up.
    raw_2014 = import_and_clean_irs("nh_2014.xlsx")
    raw_2015 = import_and_clean_irs("nh_2015.xlsx")
    raw_2016 = import_and_clean_irs("nh_2016.xlsx")
    raw_2017 = import_and_clean_irs("nh_2017.xlsx")
    raw_2018 = import_and_clean_irs("nh_2018.xlsx")
    raw_2019 = import_and_clean_irs("nh_2019.xlsx")

    # Reshape into smaller dataframes based on only ZIP.
    df_2014 = reshape_by_ZIP(raw_2014)
    df_2015 = reshape_by_ZIP(raw_2015)
    df_2016 = reshape_by_ZIP(raw_2016)
    df_2017 = reshape_by_ZIP(raw_2017)
    df_2018 = reshape_by_ZIP(raw_2018)
    df_2019 = reshape_by_ZIP(raw_2019)

    # Create 'change in' variables between years.
    df_2015 = create_deltas(df_2015, df_2014)
    df_2016 = create_deltas(df_2016, df_2015)
    df_2017 = create_deltas(df_2017, df_2016)
    df_2018 = create_deltas(df_2018, df_2017)
    df_2019 = create_deltas(df_2019, df_2018)

    # Fully normalizes entire dataframe for each year using zscores.
    # Different method from normalize_features
    df_2015_norm = stats.zscore(df_2015, axis=1)
    df_2016_norm = stats.zscore(df_2016, axis=1)
    df_2017_norm = stats.zscore(df_2017, axis=1)
    df_2018_norm = stats.zscore(df_2018, axis=1)
    df_2019_norm = stats.zscore(df_2019, axis=1)

    """
    # Print data dictionary for report.
    print(f'Data Dictionary for 2015')
    print(f'Target: Change in returns since previous year')
    print(f'Features: ')
    for col in df_2015.columns:
        print(col)
    """

    # Dimension reduction example.
    # Not used in full run
    #engineered_2015 = get_f_important_features(df_2015.drop(columns=['Change in returns since previous year']), 15, 15)
    #engineered_2015['Change in returns since previous year'] = df_2015['Change in returns since previous year']

    # Original normalization example.
    # Not used in full run
    #engineered_2015 = normalize_features(df_2015, 10)



    # FULL RUN (Will take a while.)
    
    # Each year individually.
    print("NON-NORMALIZED METHOD")
    print("Testing 2015: ")
    test_dataframe(df_2015, 10)
    print("Testing 2016: ")
    test_dataframe(df_2016, 10)
    print("Testing 2017: ")
    test_dataframe(df_2017, 10)
    print("Testing 2018: ")
    test_dataframe(df_2018, 10)
    print("Testing 2019: ")
    test_dataframe(df_2019, 10)
    
    # Combine all years.
    df_smashed = combine_all_years([df_2015, df_2016, df_2017, df_2018, df_2019])
    test_dataframe(df_smashed, 10)
    
    # Important Features:
    print('Most important features (from custom PCA method):')
    print(f'2015: {list(get_f_important_features(df_2015.drop(columns=["Change in returns since previous year"]), 5, 5))}')
    print(f'2016: {list(get_f_important_features(df_2016.drop(columns=["Change in returns since previous year"]), 5, 5))}')
    print(f'2017: {list(get_f_important_features(df_2017.drop(columns=["Change in returns since previous year"]), 5, 5))}')
    print(f'2018: {list(get_f_important_features(df_2018.drop(columns=["Change in returns since previous year"]), 5, 5))}')
    print(f'2019: {list(get_f_important_features(df_2019.drop(columns=["Change in returns since previous year"]), 5, 5))}')

    # Each year individually.
    print("FULLY NORMALIZED METHOD")
    print("Testing 2015: ")
    test_dataframe(df_2015_norm, 10)
    print("Testing 2016: ")
    test_dataframe(df_2016_norm, 10)
    print("Testing 2017: ")
    test_dataframe(df_2017_norm, 10)
    print("Testing 2018: ")
    test_dataframe(df_2018_norm, 10)
    print("Testing 2019: ")
    test_dataframe(df_2019_norm, 10)

    # Combine all years.
    df_norm_smashed = combine_all_years([df_2015_norm, df_2016_norm, df_2017_norm, df_2018_norm, df_2019_norm])
    test_dataframe(df_norm_smashed, 5)

    # Identify important features from each year.
    print('Most important features (from custom PCA method):')
    print(f'2015: {list(get_f_important_features(df_2015_norm.drop(columns=["Change in returns since previous year"]), 5, 5))}')
    print(f'2016: {list(get_f_important_features(df_2016_norm.drop(columns=["Change in returns since previous year"]), 5, 5))}')
    print(f'2017: {list(get_f_important_features(df_2017_norm.drop(columns=["Change in returns since previous year"]), 5, 5))}')
    print(f'2018: {list(get_f_important_features(df_2018_norm.drop(columns=["Change in returns since previous year"]), 5, 5))}')
    print(f'2019: {list(get_f_important_features(df_2019_norm.drop(columns=["Change in returns since previous year"]), 5, 5))}')

    # END FULL RUN


    """
    # Exporting my cleaned data.
    df_2015.to_csv('ExportedCSVs/byZIP_2015.csv', index=False)
    df_2016.to_csv('ExportedCSVs/byZIP_2016.csv', index=False)
    df_2017.to_csv('ExportedCSVs/byZIP_2017.csv', index=False)
    df_2018.to_csv('ExportedCSVs/byZIP_2018.csv', index=False)
    df_2019.to_csv('ExportedCSVs/byZIP_2019.csv', index=False)
    """

    # NOTES AND INVESTIGATION SECTION (testing stuff)

    """
    # TEST YEAR VS REDUCED.
    # Can explain much of the variance with a fraction of the features.
    important = get_f_important_features(df_2015.drop(columns=['Change in returns since previous year']), 15, 15)
    important['Change in returns since previous year'] = df_2015['Change in returns since previous year']

    avg_score_small_set = 0
    for i in range(100):
        # Does train test split inside.
        rf_score, rf_reg = create_forest(important)
        avg_score_small_set += rf_score
    avg_score_small_set /= 100
    print(f'Average score with 15 important features: {avg_score_small_set}')

    test_year(2015)
    """

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
    print(df_2016['ZIP'].nunique())
    print(df_2017['ZIP'].nunique())
    print(df_2018['ZIP'].nunique())
    print(df_2019['ZIP'].nunique())
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
