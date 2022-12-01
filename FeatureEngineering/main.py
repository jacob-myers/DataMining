import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv("DemographicData_VT - DemographicData_VT.csv")

    # https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/
    # print(df.isnull().sum())

    # Replace NA values in median_rent with the median value of median_rent
    median_rent_median = df['median_rent'].median()
    df[['median_rent']] = df[['median_rent']].fillna(value=median_rent_median)
    # print(df.isnull().sum())

    # Encode CITY and fill NA
    ordinal_encoder = OrdinalEncoder()
    df['CITY'] = ordinal_encoder.fit_transform(df[['CITY']])
    df['CITY'] = df['CITY'].astype(int)

    # Feature Engineering
    # Population count per race
    df['white_Population'] = df['total_population'] * df['percent_white'] / 100
    df['black_Population'] = df['total_population'] * df['percent_black'] / 100
    df['asian_Population'] = df['total_population'] * df['percent_asian'] / 100
    df['hispanic_Population'] = df['total_population'] * df['percent_hispanic'] / 100
    df = df.drop(columns=['percent_white', 'percent_black', 'percent_asian', 'percent_hispanic'])

    # Gross income calculated by Per Capita * population.
    df['Gross Income'] = df['Per Capita Income'] * df['total_population']

    # People per unit of area.
    df['population_density'] = df['total_population']/df['AREA']

    # DATA NORMALIZATION

    """
    # Attempt 1 - Abandoned.
    # Following code makes score much worse and volatile (many high negatives).
    
    # Normalize to range 0-1
    df['CITY'] = (df['CITY'] - df['CITY'].min()) / (df['CITY'].max() - df['CITY'].min())
    df['Zip Code'] = (df['Zip Code'] - df['Zip Code'].min()) / (df['Zip Code'].max() - df['Zip Code'].min())
    
    # Normalize with z-score
    df['total_population'] = abs((df['total_population'] - df['total_population'].mean())/(df['total_population'].std()))
    df['white_Population'] = abs((df['white_Population'] - df['white_Population'].mean())/(df['white_Population'].std()))
    df['black_Population'] = abs((df['black_Population'] - df['black_Population'].mean())/(df['black_Population'].std()))
    df['asian_Population'] = abs((df['asian_Population'] - df['asian_Population'].mean())/(df['asian_Population'].std()))
    df['hispanic_Population'] = abs((df['hispanic_Population'] - df['hispanic_Population'].mean())/(df['hispanic_Population'].std()))

    df['ALAND10'] = abs((df['ALAND10'] - df['ALAND10'].mean())/(df['ALAND10'].std()))
    df['AWATER10'] = abs((df['AWATER10'] - df['AWATER10'].mean())/(df['AWATER10'].std()))
    df['AREA'] = abs((df['AREA'] - df['AREA'].mean())/(df['AREA'].std()))
    df['Per Capita Income'] = abs((df['Per Capita Income'] - df['Per Capita Income'].mean())/(df['Per Capita Income'].std()))

    #df['median_rent'] = abs((df['median_rent'] - df['median_rent'].mean())/(df['median_rent'].std()))
    print(df.sample(10))
    """

    print(df.sample(10).to_string())

    # Normalize to range 0-1, multiply by median of target.
    # Adding median rent scale possibly improves score?
    df['CITY'] = (df['CITY'] - df['CITY'].min()) / (df['CITY'].max() - df['CITY'].min()) * df['median_rent'].median()
    df['Zip Code'] = (df['Zip Code'] - df['Zip Code'].min()) / (df['Zip Code'].max() - df['Zip Code'].min()) * df['median_rent'].median()

    # No solution found for normalization of ALAND and AWATER
    # Makes score consistently, substantially lower.
    #df['ALAND10'] = (df['ALAND10'] - df['ALAND10'].min()) / (df['ALAND10'].max() - df['ALAND10'].min()) * df['median_rent'].median()
    #df['AWATER10'] = (df['AWATER10'] - df['AWATER10'].min()) / (df['AWATER10'].max() - df['AWATER10'].min()) * df['median_rent'].median()

    # Makes score much more volatile (super high negatives, sometimes positives)
    #df['ALAND10'] = abs((df['ALAND10'] - df['ALAND10'].mean()) / (df['ALAND10'].std()))
    #df['AWATER10'] = abs((df['AWATER10'] - df['AWATER10'].mean()) / (df['AWATER10'].std()))

    # Dropping the columns altogether somehow increases score.
    df = df.drop(columns=['ALAND10', 'AWATER10'])

    """
    # Normalize population with z-score
    # Makes score substantially worse.
    df['total_population'] = abs((df['total_population'] - df['total_population'].mean()) / (df['total_population'].std()))
    df['white_Population'] = abs((df['white_Population'] - df['white_Population'].mean()) / (df['white_Population'].std()))
    df['black_Population'] = abs((df['black_Population'] - df['black_Population'].mean()) / (df['black_Population'].std()))
    df['asian_Population'] = abs((df['asian_Population'] - df['asian_Population'].mean()) / (df['asian_Population'].std()))
    df['hispanic_Population'] = abs((df['hispanic_Population'] - df['hispanic_Population'].mean()) / (df['hispanic_Population'].std()))
    """

    # Normalize with range 0 - median of target.
    # Seems to have no effect on score.
    df['total_population'] = (df['total_population'] - df['total_population'].min()) / (df['total_population'].max() - df['total_population'].min()) * df['median_rent'].median()
    df['white_Population'] = (df['white_Population'] - df['white_Population'].min()) / (df['white_Population'].max() - df['white_Population'].min()) * df['median_rent'].median()
    df['black_Population'] = (df['black_Population'] - df['black_Population'].min()) / (df['black_Population'].max() - df['black_Population'].min()) * df['median_rent'].median()
    df['asian_Population'] = (df['asian_Population'] - df['asian_Population'].min()) / (df['asian_Population'].max() - df['asian_Population'].min()) * df['median_rent'].median()
    df['hispanic_Population'] = (df['hispanic_Population'] - df['hispanic_Population'].min()) / (df['hispanic_Population'].max() - df['hispanic_Population'].min()) * df['median_rent'].median()

    # Dropping all racial data increases score for some reason.
    df = df.drop(columns=['white_Population', 'black_Population', 'asian_Population', 'hispanic_Population'])


    #print(df.sample(10).to_string())

    #df['total_population'] = abs((df['total_population'] - df['total_population'].mean()) / (df['total_population'].std()))

    # Define train test split variables.
    X = df.loc[:, df.columns != "median_age"]
    y = df['median_age']

    # Score of 100 runs.
    dt_score = 0
    for i in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Model with a decision tree regression
        dt = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=50)  # Optimized values for accuracy
        dt_reg = dt.fit(X_train, y_train)
        dt_score += dt_reg.score(X_test, y_test)
    print(dt_score/1000)
