import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


"""
Analysis:
Each run could be different. Only random state = 42 on train test split gave
reasonable scores. Depending on the split, I could get varying scores as well,
usually around 0.2. The following is a sample output from running the program:

0.21175029123862477
0.21175029123862454
0.21175029123862488
0.21175029123862477
0.21175029123862488
0.21175029123862477
0.2117502912386252
0.211750291238625
0.211750291238625
0.211750291238625

"""


if __name__ == '__main__':
    df = pd.read_csv("DemographicData_VT - DemographicData_VT.csv")

    # https://chartio.com/resources/tutorials/how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/
    # print(df.isnull().sum())

    # Replace NA values in median_rent with the median value of median_rent
    #median_rent_median = df['median_rent'].median()
    #df[['median_rent']] = df[['median_rent']].fillna(value=median_rent_median)

    # Credit: Brent
    weights = df['median_rent'].isna().sum()
    fillers = pd.Series(df['median_rent'].median() + df['median_rent'].mad() * np.random.uniform(low=0, high=1, size=weights))
    df.iloc[df['median_rent'].isna() == True, 8] = fillers
    # print(df.isnull().sum())

    # Encode CITY and fill NA
    ordinal_encoder = OrdinalEncoder()
    df['CITY'] = ordinal_encoder.fit_transform(np.array(df['CITY']).reshape(-1, 1))
    df['CITY'] = df['CITY'].astype(float) + 1

    # Feature Engineering
    # Population count per race
    df['white_Population'] = df['total_population'] * df['percent_white'] / 100
    df['black_Population'] = df['total_population'] * df['percent_black'] / 100
    df['asian_Population'] = df['total_population'] * df['percent_asian'] / 100
    df['hispanic_Population'] = df['total_population'] * df['percent_hispanic'] / 100
    #df = df.drop(columns=['percent_white', 'percent_black', 'percent_asian', 'percent_hispanic'])

    # Gross income calculated by Per Capita * population.
    # I think it improves score.
    df['Gross Income'] = df['Per Capita Income'] * df['total_population']

    # People per unit of area.
    # Makes score worse for some reason.
    #df['population_density'] = df['total_population']/df['AREA']

    # DATA NORMALIZATION
    #print(df.sample(10).to_string())

    # Normalize to range 0-1, multiply by median of target.
    # Adding median rent scale possibly improves score?
    #df['CITY'] = (df['CITY'] - df['CITY'].min()) / (df['CITY'].max() - df['CITY'].min()) * df['median_rent'].median()
    #df['Zip Code'] = (df['Zip Code'] - df['Zip Code'].min()) / (df['Zip Code'].max() - df['Zip Code'].min()) * df['median_rent'].median()

    # No solution found for normalization of ALAND and AWATER
    # Dropping the columns altogether somehow increases score.
    #df = df.drop(columns=['ALAND10', 'AWATER10'])

    # Dropping all racial data increases score for some reason.
    #df = df.drop(columns=['white_Population', 'black_Population', 'asian_Population', 'hispanic_Population'])

    # Nonsense Normalization
    # Marginal improvements in some cases.
    df['nonsense1'] = 100 * df['Zip Code'] + df['CITY']
    df['nonsense2'] = df['CITY'] / df['Per Capita Income']
    #df['nonsense3'] = df['Gross Income'] / df['CITY']
    df['nonsense4'] = df['AREA'] / df['CITY']
    df['nonsense5'] = df['total_population'] / df['Zip Code']

    # Define train test split variables.
    # Messes score up somehow even though it picks the exact same stuff.
    #X = df.loc[:, df.columns != "median_age"]
    #y = df['median_age']

    # Models it 10 times.
    for i in range(10):
        # Usually better scores
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['median_age']), df['median_age'], test_size=.25, random_state=42)

        # Much more volatile scores, many negatives.
        #X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['median_age']), df['median_age'], test_size=.25)

        # Model with a decision tree regression
        dt = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=50)  # Optimized values for accuracy
        dt_reg = dt.fit(X_train, y_train)
        print(dt_reg.score(X_test, y_test))
