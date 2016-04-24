# -*- coding: utf-8 -*-
"""
Kaggle_santander competition 

Xgboost models 

"""

import pandas as pd
import numpy as np

from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v1.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

##############################################################################
#     perform cross validation with resampling
##############################################################################

#to fix the split, use: random_state = 104
result = []
for i in range(0,25):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
    clf1 = xgb.XGBClassifier(max_depth=5, n_estimators=400, learning_rate=0.05, seed = 104, silent = True)
    clf1.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=[(X_test, y_test)], verbose = False)
    result.append(clf1.best_score)
    print(clf1.best_score)

print('---average score is: %s--- ') %np.average(result)


#Best validation auc score average is: 0.82868548, 0.831505, 0.8284052
#Let's see the leaderboard score by a submission

##################
'''
---- Submission ----
    xgb_v0.csv 

'''
submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=5, n_estimators=400, learning_rate=0.05, silent = False, nthread = 4, seed = 104)
model.fit(X, y)
prediction = model.predict_proba(test_X)


submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0.csv', index = False, )

# Leaderboard score: 0.837787, so we need to fine-tune the CV metric...

##############################################################################
#     perform k-fold cross validation
##############################################################################

from sklearn.cross_validation import StratifiedKFold

skf = StratifiedKFold(train.TARGET, n_folds= 3)
model = xgb.XGBClassifier(max_depth=5, n_estimators = 400, learning_rate=0.05, silent = True, nthread = 4, seed = 104)

result = []
for train_index, test_index in skf:
    k_train = train.drop(test_index, axis = 0) 
    k_test = train.drop(train_index, axis = 0)
    model.fit(k_train.iloc[:,:-1], k_train.iloc[:,-1], eval_metric="auc")
    prediction = model.predict_proba(k_test.iloc[:,:-1])
    score = roc_auc_score(y_true = np.array(k_test.iloc[:,-1]), y_score = prediction[:,1])
    print(score)    
    result.append(score)
np.average(result)


#k = 5: 0.79905666
#k = 10: 0.7862909576
#k = 3: 0.829418394  <- I'm gonna stick with this option,LB diff: 0.008368
#k = 2: 0.82078981

#Try out different parameters 
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold

skf = StratifiedKFold(train.TARGET, n_folds= 3)
model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 104)

result = []
for train_index, test_index in skf:
    k_train = train.drop(test_index, axis = 0) 
    k_test = train.drop(train_index, axis = 0)
    #For early_stopping it needs evaluation sets, here is the split:
    X_fit, X_eval, y_fit, y_eval= train_test_split(k_train.iloc[:,:-1], k_train.iloc[:,-1], test_size=0.3)
    #Here is the modified fitting function with early_stopping implemented
    model.fit(k_train.iloc[:,:-1], k_train.iloc[:,-1], early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
    prediction = model.predict_proba(k_test.iloc[:,:-1])
    score = roc_auc_score(y_true = np.array(k_test.iloc[:,-1]), y_score = prediction[:,1])
    print(score)
    result.append(score)
np.average(result)

#3 fold CV result is: 0.741432, but LB score is a bit better... be careful
#with this model
#It might be, that the test_size = 0.3 lead into this bad CV result... 
#Let's check it out with the value of 0.1


skf = StratifiedKFold(train.TARGET, n_folds= 3)
model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 104)

result = []
for train_index, test_index in skf:
    k_train = train.drop(test_index, axis = 0) 
    k_test = train.drop(train_index, axis = 0)
    #For early_stopping it needs evaluation sets, here is the split:
    X_fit, X_eval, y_fit, y_eval= train_test_split(k_train.iloc[:,:-1], k_train.iloc[:,-1], test_size=0.1)
    #Here is the modified fitting function with early_stopping implemented
    model.fit(k_train.iloc[:,:-1], k_train.iloc[:,-1], early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
    prediction = model.predict_proba(k_test.iloc[:,:-1])
    score = roc_auc_score(y_true = np.array(k_test.iloc[:,-1]), y_score = prediction[:,1])
    print(score)
    result.append(score)
np.average(result)
#CV result: 0.8317097

##################
'''
---- Submission ----
    xgb_v0_1.csv 

'''
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v1.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.3)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)n_components = 13


submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_1.csv', index = False, )

#-----------------
# Leaderboard score: 0.838791 a little bit better... 
#!!!! Unfortunately I did not fixed the random number for train_test_split, therefore I can't reproduce the result... 
##################

'''
---- Submission ----
    xgb_v0_2.csv 

'''
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v1.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.1)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)


submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_2.csv', index = False, )

#-----------------
# Leaderboard score: 0.838791, which is the same as before... strange :)
#!!!! Unfortunately I did not fixed the random number for train_test_split, therefore I can't reproduce the error... 
##################

'''
---- Submission ----
    xgb_v0_11.csv <- like v0.1, but wo ID column

'''
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v1.csv")
X = train.iloc[:,1:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,1:-1]
test_y = test.iloc[:,-1]


submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.3)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)


submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_11.csv', index = False, )

#-----------------
# Leaderboard score: 0.822124 , which is not an improvement
#!!!! Unfortunately I did not fixed the random number for train_test_split, therefore I can't reproduce the error... 
##################

'''
---- Submission ----
    xgb_v0_21.csv <- like v0.2, but wo ID column

'''

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v1.csv")
X = train.iloc[:,1:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,1:-1]
test_y = test.iloc[:,-1]

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.1)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)


submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_21.csv', index = False, )

#-----------------
#Leaderboard result is: 0.823149
##################

'''
---- Submission ----
    xgb_v0_1FE2.csv 
    Which is xgb_v0_1, with FE_v2 train/test

'''
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v2.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v2.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.3)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_1FE2_fixedSeed.csv', index = False, )

#-----------------
#Leaderboard score: 0.839407 Which is a small improvement.
#-----------------

for i in range(0,10):
    model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4)
    X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.3)
    model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
    prediction += model.predict_proba(test_X)

prediction = prediction/11

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_1FE2.csv', index = False, )

#-----------------
''' 
Leaderboard score: 0.823896 LOOOOL, which means, that it's not an 
Which means that this is not an improvement.
'''
#-----------------

'''
---- Submission ----
    xgb_v0_1FE3.csv 
    Which is xgb_v0_1, with FE_v2 train/test

'''
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v3.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v3.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.3)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_1FE3_fixedSeed.csv', index = False, )

#validation_score = 0.879323

#-----------------
#Leaderboard score: 0.838858, which is slightly worse....  NOT an improvement
#-----------------

'''
---- Submission ----
    xgb_v0_1FE4_1.csv 
    Which is xgb_v0_1, with FE_v4.1 train/test

'''
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v4_1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v4_1.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=6, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.3)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_1FE4_1.csv', index = False, )

#-----------------
#Leaderboard score: Tiny improvement: 0.839494 by 0.000087
#For now, this is the best solution
#-----------------

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v4_1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v4_1.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]
submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=6, n_estimators = 500, learning_rate=0.02, subsample = 1, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.1)
model.fit(X, y)
prediction = model.predict_proba(test_X)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_3FE4_1.csv', index = False, )
#-----------------
#Leaderboard score: 0.838544 which is not an improvement. 
#-----------------

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v4_3.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v4_3.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]
submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=6, n_estimators = 500, learning_rate=0.02, subsample = 1, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.1)
model.fit(X, y)
prediction = model.predict_proba(test_X)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_3FE4_3.csv', index = False, )

#-----------------
#Leaderboard score:  0.838549, which is not an improvement...
#-----------------

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v4_2.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v4_2.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]
submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=6, n_estimators = 500, learning_rate=0.02, subsample = 1, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.1)
model.fit(X, y)
prediction = model.predict_proba(test_X)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_3FE4_2.csv', index = False, )

#-----------------
#Leaderboard score: 0.838695, which is not an improvement, but slightly better, than the previous one for the same version of XGB. Let's try out the best version of XGB sofar: 
#-----------------
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v4_2.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v4_2.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=6, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.3)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_1FE4_2.csv', index = False, )

#-----------------
#Leaderboard score: 0.839246, which is not an improvement...
#-----------------
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v4_3.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v4_3.csv")
X = train.iloc[:,:-1]    #Explanatory variables for model fitting
y = train.iloc[:,-1]    #TARGET variable
test_X = test.iloc[:,:-1]
test_y = test.iloc[:,-1]

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=6, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(X, y, test_size=0.3)
model.fit(X, y, early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(test_X)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_1FE4_3.csv', index = False, )

#-----------------
#Leaderboard score: 0.838696, which is not an improvement... 
#-----------------

