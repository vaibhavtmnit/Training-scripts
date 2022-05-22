import pandas as pd
import numpy as np
from path import Path

import xgboost as xgb


# Training XGBoost
dtrain = xgb.DMatrix(data = x_train, label = y_train)
dval = xgb.DMatrix(data = x_val, label = y_val)

params_tune = {
    
    ### Refer Page : https://xgboost.readthedocs.io/en/stable/parameter.html ###
    
    #### TREE PARAM ####
    'booster': 'gbtree',
    
    #### BOOSTER PARAM ####
    'learning_rate': .1,
    # gamma is reduction in loss require dto split a leaf
    'gamma': 0, 
    'max_depth': 4,  
    # min_child_weight is sum of total weight required to further split the node
    'min_child_weight': 1, 
    'subsample': .8,
    'colsample_bytree': .9,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'lambda': 1,
    'alpha': 1,
    'tree_method': 'exact',
    # process_type is used to update an existing model, default value creates new model
    'process_type':'default',
    # max_bin : these many bins are created to bucket cont. features, doesn't work with exact tree_method
    'max_bin': 256,
    
    #### LEARNING PARAM ####
    
    'objective': 'binary:logistic',
    # Initial Prediction for all instances, global bias
    'base_score': 0.5,
    # Evaluation metric for valuation data
    # combinations of metrics is way to go, e.g. loggloss will show is loss is decresing and error at the same time
    # will tell if focus is on wrong or correct prediction and auc will show both pr and recall are getting better or not
    # 
    'eval_metric': ['logloss','error','auc'],
    # Seed for reproducibility
    'seed':0    
    
}

train_params = {
    
    # feval is deprecated since  v1.6.0
    'num_boost_round':120,
    # Don't use with optuna
    'early_stopping_rounds': 20,
    # custom_metric: '', Custom metric needs to be defined here instead of tuning paramaeters dictionary
    
    }

# For more features refer below links
## Feature Interaction Constraints : https://xgboost.readthedocs.io/en/stable/tutorials/feature_interaction_constraint.html
## Callbacks : https://xgboost.readthedocs.io/en/stable/python/callbacks.html
## https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.callback
## Monotonic Constraints : https://xgboost.readthedocs.io/en/stable/tutorials/monotonic.html
## Custom objective function : https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html


# Training a simple xgboost model with above parameters
xgb_model = xgb.train(params_tune,dtrain, evals=[(dval, 'eval'), (dtrain, 'train')],**train_params)

# Tranining model using xgboost's in-built cv
X = pd.concat([x_train,x_val],axis=0).reset_index(drop=True)
y = pd.concat([y_train,y_val],axis=0).reset_index(drop=True)
DMat = xgb.DMatrix(X,label=y)
xgb.cv(params_tune,DMat,120,nfold=5,early_stopping_rounds = 25)
