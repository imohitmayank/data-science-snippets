"""Hyperparameter tuning for SVR regression algorithm

- Specify the search space i.e. the list of algorithm parameters to try
- for each parameter combination perform a 5 fold CV test

Algorithm page: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
"""

# import =======
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error

# DATA LOAD ============
train_data = ...  # load the features and target on which to train

# SEARCH SPACE ============
search_space = [{'kernel': ['poly', 'rbf', 'sigmoid'],
               'C': [1, 10, 100], 'epsilon': [10, 1, 0.1, 0.2, 0.01]}]

# TUNING ============
scorer = make_scorer(mean_squared_error, greater_is_better=False)
svr_gs = GridSearchCV(SVR(), search_space, cv = 5, scoring=scorer, verbose=10, n_jobs=None)
svr_gs.fit(train_data['features'], train_data['target'])


# PRINT RESULT ============
parameter_result = []
print("Grid scores on training set:")
means = svr_gs.cv_results_['mean_test_score']
stds = svr_gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, svr_gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
    parameter_result.append({'mean': abs(mean), 'std': std, **params})
    
# SELECT BEST PARAMETERS ============
# select the settings with smallest loss
parameter_result = pd.DataFrame(parameter_result)
parameter_result = parameter_result.sort_values(by=['mean'])
best_settings = parameter_result.head(1).to_dict(orient='records')[0]

# FIT WITH BEST PARAMETERS ============
SVRModel = SVR(C=best_settings['C'], 
            epsilon=best_settings['epsilon'], 
            kernel= best_settings['kernel'])
SVRModel.fit(train_data['features'], train_data['target'])
