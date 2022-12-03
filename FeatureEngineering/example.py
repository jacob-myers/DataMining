import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree

data = pd.read_csv('DemographicData_VT - DemographicData_VT.csv')

#Data Engineering
#Encode CITY and Fill NA's
weights = data['median_rent'].isna().sum()
fillers = pd.Series(data['median_rent'].median()+data['median_rent'].mad()*np.random.uniform(low=0, high=1, size=weights))
data.iloc[data['median_rent'].isna()==True, 8]=fillers
#data['median_rent'].fillna(value=data['median_rent'].median(), inplace=True)

#Feature Engineering
data['white_population']=data['total_population']*data['percent_white']/100
data['black_population']=data['total_population']*data['percent_black']/100
data['hispanic_population']=data['total_population']*data['percent_hispanic']/100
data['asian_population']=data['total_population']*data['percent_asian']/100

#Setup simple DecisionTreeRegressor
enc = OrdinalEncoder()
data['CITY'] = enc.fit_transform(np.array(data['CITY']).reshape(-1,1))


#Model
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['median_age']), data['median_age'], test_size=.25, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(data[['asian_population']], data['median_age'], test_size=.25, random_state=42)

trgr = tree.DecisionTreeRegressor(max_depth=5, max_leaf_nodes=50)
trgr_fit = trgr.fit(x_train, y_train)
print(trgr_fit.score(x_test, y_test))