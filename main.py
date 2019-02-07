from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
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
print('=====\nTrain against different learning algorithms and pick one that produces the best result\n=====')
model_factory = [
    RandomForestRegressor(),
    # XGBRegressor(nthread=1),
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

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=num_folds,
        scoring='neg_mean_squared_error'
    )
    score_description = " %0.2f (+/- %0.2f)" % (
        np.sqrt(scores.mean()*-1), scores.std() * 2)

    print('{model:25} CV-5 RMSE: {score}'.format(
        model=model.__class__.__name__,
        score=score_description
    ))


# Create Pseudo-labeler class


class PseudoLabeler(BaseEstimator, RegressorMixin):
    '''
    Sci-kit learn wrapper for creating pseudo-lebeled estimators.
    '''

    def __init__(self, model, unlabeled_data, features, target, sample_rate=0.2, seed=42):
        '''
        @sample_rate - percent of samples used as pseudo-labelled data
                       from the unlabled dataset
        '''
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'

        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        self.unlabeled_data = unlabeled_data
        self.features = features
        self.target = target

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabeled_data": self.unlabeled_data,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        '''
        Fit the data using pseudo labeling.
        '''
        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
            augemented_train[self.features],
            augemented_train[self.target]
        )
        return self

    def __create_augmented_train(self, X, y):
        '''
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        '''
        num_of_samples = int(len(self.unlabeled_data) * self.sample_rate)

        # Train the model and create the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabeled_data[self.features])

        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabeled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)

    def predict(self, X):
        '''
        Returns the predicted values.
        '''
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__

    def get_feature_importance(self):
        return self.model.feature_importances_


print('\n\n=====\nUse Gradient Boosting Regressor to find a good sampling rate for Pseudo Labelling\n=====')

sample_rates = np.linspace(0, 1, 10)


def pseudo_label_wrapper(model):
    return PseudoLabeler(model, test, features, target)


# List of all models to test
model_factory = [
    RandomForestRegressor(n_jobs=1),
    GradientBoostingRegressor(),
]

# Apply the PseudoLabeler class to each model
model_factory = map(pseudo_label_wrapper, model_factory)

# Train each model with different sample rates
results = {}
num_folds = 5

for model in model_factory:
    model_name = model.get_model_name()
    print('%s' % model_name)

    results[model_name] = list()
    for sample_rate in sample_rates:
        model.sample_rate = sample_rate

        # Calculate the CV-3 R2 score and store it
        scores = cross_val_score(
            model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error', n_jobs=8)
        results[model_name].append(np.sqrt(scores.mean()*-1))

plt.figure(figsize=(16, 18))

# i = 1
# for model_name, performance in results.items():
#     plt.subplot(3, 3, i)
#     i += 1

#     plt.plot(sample_rates, performance)
#     plt.title(model_name)
#     plt.xlabel('sample_rate')
#     plt.ylabel('RMSE')

# plt.show()


'''
params = {
    'n_estimators': 500,
    'max_depth': 4,
    'min_samples_split': 2,
    'learning_rate': 0.01,
    'loss': 'ls'
}
model_factory = [
    GradientBoostingRegressor(**params),
    PseudoLabeler(
        GradientBoostingRegressor(**params),
        test,
        features,
        target,
        sample_rate=1.0
    ),
]

for model in model_factory:
    model.seed = 42
    num_folds = 8

    scores = cross_val_score(
        model, X_train, y_train, cv=num_folds, scoring='neg_mean_squared_error', n_jobs=8)
    score_description = "MSE: %0.4f (+/- %0.4f)" % (
        np.sqrt(scores.mean()*-1), scores.std() * 2)

    print('{model:25} CV-{num_folds} {score_cv}'.format(
        model=model.__class__.__name__,
        num_folds=num_folds,
        score_cv=score_description
    ))
'''

print('\n\n=====\nFinalized, tuned gradient boosting regressor with pseudo-labelling\n=====')

# tuned model
params = {
    'n_estimators': 500,
    'max_depth': 4,
    'min_samples_split': 2,
    'learning_rate': 0.01,
    'loss': 'ls'
}
num_folds = 3

model = PseudoLabeler(
    GradientBoostingRegressor(**params),
    test,
    features,
    target,
    sample_rate=0.79
)
# model = GradientBoostingRegressor(**params)
model.fit(X_train, y_train)
pred = model.predict(X_test)

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=num_folds,
    scoring='neg_mean_squared_error',
    n_jobs=8
)

score_description = "MSE: %0.4f (+/- %0.4f)" % (
    np.sqrt(scores.mean()*-1), scores.std() * 2)
print('{model:25} CV-{num_folds} {score_cv}'.format(
    model=model.__class__.__name__,
    num_folds=num_folds,
    score_cv=score_description
))

# Plot feature importance
feature_importance = model.get_params().get('model').feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, features[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
