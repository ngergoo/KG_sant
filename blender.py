# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 21:31:09 2016

@author: ngergoo
"""

##########
#Blend together xgb_v0, and xgb.v0_2
#Because they look really different

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

submission1 = pd.read_csv('Documents/DataMining/Santander/Submission/xgb_v0.csv')
submission2 = pd.read_csv('Documents/DataMining/Santander/Submission/xgb_v0_1.csv')

#Check the plot to see the difference
plt.plot(submission1.TARGET,'o')
plt.plot(submission2.TARGET,'o')


##################
'''
---- Submission ----
    blend_v1.csv 

'''
submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

submission.TARGET = (submission1.TARGET + submission2.TARGET)/2
pd.DataFrame.to_csv(submission, 'blend_v1.csv', index = False, )

plt.plot(submission.TARGET, 'o')

#-----------------
#Result is: 0.837068, which is not an improvement. 

##################
'''
---- Submission ----
    blend_v2.csv 
Blend together the best xgb(xgb_v0_1FE2_fixedSeed.csv) and RF ( RF_v0_1.csv ) sofar
'''
submission_xgb = pd.read_csv('Documents/DataMining/Santander/Submission/xgb_v0_1FE2_fixedSeed.csv')
submission_rf = pd.read_csv('Documents/DataMining/Santander/Submission/RF_v0_1.csv')

plt.plot(submission_xgb.TARGET, 'o')
plt.plot(submission_rf.TARGET, 'o')

#50-50% weight
submission.TARGET = (submission_rf.TARGET + submission_xgb.TARGET)/2
pd.DataFrame.to_csv(submission, 'blend_v2.csv', index = False, )
#-----------------
#Leaderboard score: 0.827377, which is not an improvement....
#-----------------

#25-75% weight in favour of xgb. 
submission.TARGET = (0.25 * submission_rf.TARGET + 0.75 * submission_xgb.TARGET)
pd.DataFrame.to_csv(submission, 'blend_v2_1.csv', index = False, )
#-----------------
#Leaderboard score:  0.836083 Which is not an improvement....
#-----------------

##################
'''
---- Submission ---- 
    blend_v3_1.csv
Blend together the best and the 2nd best xgb sofar: 
xgb_v0_1FE4_1.csv AND 
xgb_v0_3FE4_2.csv
'''

#50-50% weight in facour of xgb. 
submission_1 = pd.read_csv('Documents/DataMining/Santander/Submission/xgb_v0_1FE4_1.csv')
submission_2 = pd.read_csv('Documents/DataMining/Santander/Submission/xgb_v0_3FE4_2.csv')

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

submission.TARGET = (submission_1.TARGET + submission_2.TARGET)/2
pd.DataFrame.to_csv(submission, 'blend_v3_1.csv', index = False, )

#-----------------
#Leaderboard score: 0.839493 which is not an improvement (slightly worse)
#-----------------
'''
---- Submission ---- 
    blend_v4_1.csv
Blend together the following predictions: 
- xgb_r_v1_FEv4_3, LB Score: 0.839968
- xgb_v0_1FE4_1, LB Score: 0.839494


'''

#50-50% weight in facour of xgb. 
submission_1 = pd.read_csv('Documents/DataMining/Santander/Submission/xgb_r_v1_FEv4_3.csv')

submission_1.TARGET = 1-submission_1.TARGET

submission_2 = pd.read_csv('Documents/DataMining/Santander/Submission/xgb_v0_1FE4_1.csv')

submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

submission.TARGET = (submission_1.TARGET + submission_2.TARGET)/2
pd.DataFrame.to_csv(submission, 'blend_v4_1.csv', index = False, )

#-----------------
#Leaderboard score: 0.839822 which is not an improvement (slightly worse)
#-----------------



