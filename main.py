from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('./Train.csv')
test = pd.read_csv('./Test.csv')

# features to keep
fields = [
    'Item_Weight',
    'Item_Fat_Content',
    'Outlet_Establishment_Year',
    'Outlet_Size',
    'Item_Visibility',
    'Item_MRP',
    'Outlet_Location_Type',
    'Outlet_Type',
    'Item_Outlet_Sales'
]

train = train[train.columns.intersection(fields)]
test = test[test.columns.intersection(fields)]

# impute mean
train['Item_Weight'].fillna((train['Item_Weight'].mean()), inplace=True)
test['Item_Weight'].fillna((test['Item_Weight'].mean()), inplace=True)

# reduce fat content to two categories
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(
    ['low fat', 'LF'], ['Low Fat', 'Low Fat'])
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['reg'], [
                                                              'Regular'])
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(
    ['low fat', 'LF'], ['Low Fat', 'Low Fat'])
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['reg'], [
                                                            'Regular'])

# calculate establishment year
train['Outlet_Establishment_Year'] = 2013 - train['Outlet_Establishment_Year']
test['Outlet_Establishment_Year'] = 2013 - test['Outlet_Establishment_Year']
# default small for missing outlet_size
train['Outlet_Size'].fillna('Small', inplace=True)
test['Outlet_Size'].fillna('Small', inplace=True)

# label encoding cate. var.
col = ['Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type', 'Item_Fat_Content']
test['Item_Outlet_Sales'] = 0

combi = train.append(test)
number = LabelEncoder()
for i in col:
    combi[i] = number.fit_transform(combi[i].astype('str'))
    combi[i] = combi[i].astype('int')
train = combi[:train.shape[0]]
test = combi[train.shape[0]:]
test = test.drop('Item_Outlet_Sales', axis=1)

# set features and labels
y_train = train['Item_Outlet_Sales']
train = train.drop('Item_Outlet_Sales', axis=1)

features = train.columns
target = 'Item_Outlet_Sales'

X_train, X_test = train, test

# Starting with different supervised learning algorithm, let us check which algorithm gives us the best results

#from sklearn.neural_network import MLPRegressor


model_factory = [
    RandomForestRegressor(),
    XGBRegressor(nthread=1),
    # MLPRegressor(),
    Ridge(),
    BayesianRidge(),
    ExtraTreesRegressor(),
    ElasticNet(),
    KNeighborsRegressor(),
    GradientBoostingRegressor()
]

for model in model_factory:
    model.seed = 42
    num_folds = 3

scores = cross_val_score(model, X_train, y_train,
                         cv=num_folds, scoring='neg_mean_squared_error')
score_description = " %0.2f (+/- %0.2f)" % (
    np.sqrt(scores.mean()*-1), scores.std() * 2)

print('{model:25} CV-5 RMSE: {score}'.format(
    model=model.__class__.__name__,
    score=score_description
))
