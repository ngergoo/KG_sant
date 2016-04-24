# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import gc

"""
Created on Sat Apr  2 19:32:58 2016
@author: ngergoo

Feature engineering
Baseline is FE_v2, with the best LB result. 

FE_v2 contains: 
    - drop of explanatory variables with 0 std. 
    - removed duplicated variables. 
    
FE_v3 was not successful,
    - it is one more round of variable removal regarding highly correlated 
        variables. 

FE_v4
Check variables with potentially categorical infos, or with very low values
Check for variables with high values

Calculate mean/median/sum scores per row
"""

#Load starting point_FE_v2 files
train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v2.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v2.csv")


#Create combi table for further analysis
combi = pd.concat([train, test], axis = 0)
#-----------------------------------------------------------------------------
'''
I might fucked this up, because of the combi table. 
I've calculated min/max/std/mean ect scores for the combi table, and not for 
the train, and test table... 
'''
#-----------------------------------------------------------------------------

#To devide combi table to train and test use this
#train = combi.iloc[0:76020,:]
#test = combi.iloc[76020:,:]


#gather data for low std variables
temp = pd.DataFrame()

for i in range(0,len(combi.columns)):
    if combi.iloc[:,i].std() < 5: 
        temp = pd.concat([temp, combi.iloc[:, i]], axis = 1)

temp = temp.drop('TARGET', axis = 1)

#167 columns have very low stds, calculate everything for them

combi['low_sum'] = temp.sum(axis = 1)
combi['low_mean'] = temp.mean(axis = 1)
combi['low_median'] = temp.median(axis = 1)

drop(temp)
gc.collect()

#gather data for high std variables

temp = pd.DataFrame()

for i in range(0,len(combi.columns)):
    if combi.iloc[:,i].std() >= 5: 
        temp = pd.concat([temp, combi.iloc[:, i]], axis = 1)

combi['high_sum'] = temp.sum(axis = 1)
combi['high_mean'] = temp.mean(axis = 1)
combi['high_median'] = temp.median(axis = 1)
combi['high_min'] = temp.min(axis = 1)
combi['high_max'] = temp.max(axis = 1)


#Write out results
target = pd.DataFrame(combi.TARGET)
combi = combi.drop('TARGET', axis = 1)
combi = pd.concat([combi, target], axis = 1)

train = combi.iloc[0:76020,:]
test = combi.iloc[76020:,:]

pd.DataFrame.to_csv(train, 'train_FE_v4_1.csv', index = False)
pd.DataFrame.to_csv(test, 'test_FE_v4_1.csv', index = False)

#-----------------------------------------------------------------------------
# Feature engineering v4.2. 
# Fix for calculating everything from the combi table

#Add standard deviation of low std and high std columns
#Add low_min, low_max calculations


train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v4_1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v4_1.csv")

#save target columns
target_train = pd.DataFrame(train.TARGET)
target_test = pd.DataFrame(test.TARGET)

#drop calculations of the v4.1, and TARGET
train = train.drop(['low_sum', 'high_sum', 'high_mean', 'high_min', 'high_max', 'TARGET'], axis = 1)
test = test.drop(['low_sum', 'high_sum', 'high_mean', 'high_min', 'high_max', 'TARGET'], axis = 1)

combi = pd.concat([train, test], axis = 0)

temp = pd.DataFrame()

#Calculate columns with low std, because we need to aggregate these
#differently... 

for i in range(0,len(combi.columns)):
    if combi.iloc[:,i].std() >= 5: 
        temp = pd.concat([temp, combi.iloc[:, i]], axis = 1)

#Save tables for calculations: 
train_low = train.loc[:,temp.columns]
train_high = train.drop(temp.columns, axis = 1)
test_low = test.loc[:,temp.columns]
test_high = test.drop(temp.columns, axis = 1)

train.loc[:,'low_sum'] = train_low.sum(axis = 1) 
train.loc[:,'low_mean'] = train_low.mean(axis = 1) 
train.loc[:,'low_median'] = train_low.median(axis = 1) 

train.loc[:,'high_sum'] = train_high.sum(axis = 1) 
train.loc[:,'high_mean'] = train_high.mean(axis = 1) 
train.loc[:,'high_median'] = train_high.median(axis = 1) 
train.loc[:,'high_min'] = train_high.min(axis = 1) 
train.loc[:,'high_max'] = train_high.max(axis = 1) 


test.loc[:,'low_sum'] = test_low.sum(axis = 1) 
test.loc[:,'low_mean'] = test_low.mean(axis = 1) 
test.loc[:,'low_median'] = test_low.median(axis = 1) 

test.loc[:,'high_sum'] = test_high.sum(axis = 1) 
test.loc[:,'high_mean'] = test_high.mean(axis = 1) 
test.loc[:,'high_median'] = test_high.median(axis = 1) 
test.loc[:,'high_min'] = test_high.min(axis = 1) 
test.loc[:,'high_max'] = test_high.max(axis = 1) 

#Add TARGET column back 
train = pd.concat([train, target_train], axis = 1)
test = pd.concat([test, target_test], axis = 1)

#Write out completed dataframes
pd.DataFrame.to_csv(train, 'train_FE_v4_2.csv', index = False)
pd.DataFrame.to_csv(test, 'test_FE_v4_2.csv', index = False)

#-----------------------------------
# Feature engineering v4.3 

#Add standard deviation of low std and high std columns
#Add low_min, low_max calculations

target_train = pd.DataFrame(train.TARGET)
target_test = pd.DataFrame(test.TARGET)

train = train.drop('TARGET', axis = 1)
test = test.drop('TARGET', axis = 1)

train.loc[:,'low_min'] = train_low.min(axis = 1) 
train.loc[:,'low_max'] = train_low.max(axis = 1) 
train.loc[:,'low_std'] = train_low.std(axis = 1)
train.loc[:,'high_std'] = train_high.std(axis = 1)

test.loc[:,'low_min'] = test_low.min(axis = 1) 
test.loc[:,'low_max'] = test_low.max(axis = 1)
test.loc[:,'low_std'] = test_low.std(axis = 1)
test.loc[:,'high_std'] = test_high.std(axis = 1)

#Add TARGET column back 
train = pd.concat([train, target_train], axis = 1)
test = pd.concat([test, target_test], axis = 1)

#Write out completed dataframes
pd.DataFrame.to_csv(train, 'train_FE_v4_3.csv', index = False)
pd.DataFrame.to_csv(test, 'test_FE_v4_3.csv', index = False)


#-----------------------------------




