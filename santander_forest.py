# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:32:11 2016

@author: ngergoo
"""

import pandas as pd
import numpy as np
from sklearn import ensemble 
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v2.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v2.csv")



skf = StratifiedKFold(train.TARGET, n_folds= 5)
model = ensemble.RandomForestClassifier(n_estimators = 10, oob_score = True, verbose = 1, random_state = 104, n_jobs = 4)

result = []
for train_index, test_index in skf:
    k_train = train.drop(test_index, axis = 0) 
    k_test = train.drop(train_index, axis = 0)
    model.fit(k_train.iloc[:,:-1], k_train.iloc[:,-1])
    prediction = model.predict_proba(k_test.iloc[:,:-1])
    score = roc_auc_score(y_true = np.array(k_test.iloc[:,-1]), y_score = prediction[:,1])
    print(score)    
    result.append(score)
np.average(result)
#Average result: 0.68844939

basz = pd.DataFrame(model.feature_importances_)
basz.columns = ['varimp']
basz.sort('varimp', inplace = True)
#First 93 variables are useless. 
train = train.drop(train.columns[basz.index[:93]], axis = 1)
test = test.drop(test.columns[basz.index[:93]], axis = 1)

#Run CV again
result = []
for train_index, test_index in skf:
    k_train = train.drop(test_index, axis = 0) 
    k_test = train.drop(train_index, axis = 0)
    model.fit(k_train.iloc[:,:-1], k_train.iloc[:,-1])
    prediction = model.predict_proba(k_test.iloc[:,:-1])
    score = roc_auc_score(y_true = np.array(k_test.iloc[:,-1]), y_score = prediction[:,1])
    print(score)    
    result.append(score)
np.average(result)
#Average result: 0.6992919

model = ensemble.RandomForestClassifier(n_estimators = 750, oob_score = True, verbose = 1, random_state = 104, n_jobs = 4)

#Run CV again
result = []
for train_index, test_index in skf:
    k_train = train.drop(test_index, axis = 0) 
    k_test = train.drop(train_index, axis = 0)
    model.fit(k_train.iloc[:,:-1], k_train.iloc[:,-1])
    prediction = model.predict_proba(k_test.iloc[:,:-1])
    score = roc_auc_score(y_true = np.array(k_test.iloc[:,-1]), y_score = prediction[:,1])
    print(score)    
    result.append(score)
np.average(result)
#Average result: 0.76461440620635812

#Write out train and test table for later use. 
pd.DataFrame.to_csv(train, 'train_FE_v2_RF_varimp.csv', index = False)
pd.DataFrame.to_csv(test, 'test_FE_v2_RF_varimp.csv', index = False)


##################
'''
---- Submission ----
    RF_v0_1.csv 

'''
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v2_RF_varimp.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v2_RF_varimp.csv")

model = ensemble.RandomForestClassifier(n_estimators = 750, oob_score = True, verbose = 1, random_state = 104, n_jobs = 4)
model.fit(train.iloc[:,:-1], train.iloc[:,-1])
prediction = model.predict_proba(test.iloc[:,:-1])

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")
submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'RF_v0_1.csv', index = False, )

#Leaderboard score is: 0.789739 which is of course not an improvement, but higher than the CV value...
from matplotlib import pyplot as plt
plt.plot(submission.TARGET, 'o') 

