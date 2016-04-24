# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/ngergoo/.spyder2/.temp.py
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing

train = pd.read_csv('Documents/DataMining/Santander/CleanData/train.csv')
test = pd.read_csv('Documents/DataMining/Santander/CleanData/test.csv')
submission = pd.read_csv('Documents/DataMining/Santander/CleanData/sample_submission.csv')


##############################################################################
#               EXPLANATORY VARIABLE TESTS
##############################################################################
#Create combi table
test['TARGET'] = 0
combi = pd.concat([train, test], axis = 0)


#rename variables of train dataset to v1, v2, etc
a = []
for i in range(0,370):
    a.append(str('v')+str(i))
a.append('TARGET')
combi.columns = a

variables = pd.DataFrame(combi.columns)
variables.columns = ['column_name']

#Check min, max, distinct_count, sd
#For every row, summarized in the variables table
variables['min'] = 0
variables['max'] = 0
variables['distinct_count'] = 0
variables['sd'] = 0
variables['na_count'] = 0


#Calculate values for each columns: 
for i in range(0,len(combi.columns)):
    variables.iloc[i,1] = combi.iloc[:,i].min()
    variables.iloc[i,2] = combi.iloc[:,i].max()
    variables.iloc[i,3] = len(combi.iloc[:,i].unique())
    variables.iloc[i,4] = np.std(combi.iloc[:,i])
    variables.iloc[i,5] = sum(pd.isnull(combi.iloc[:,i]))
    if i % 75 == 0:
        print(i)
        
len(variables.loc[variables.sd == 0])
        
"""
Conclusions: 
    - 34 variables have 0 standard deviation. Values are 0 there. 
    - There are no missing values in the dataframe
    - v1 variable has weird values, min = -999999.00
"""

#Drop variables with 0 sd: 
temp = pd.DataFrame(variables.column_name.loc[variables.sd == 0])
temp = temp.reset_index(drop = True)
for name in temp.column_name:
    combi = combi.drop(name, axis = 1)

len(combi.columns)
#335 explanatory variable left, last column is target, first column is ID

###############

#Check column values again, and add type variable
variables = pd.DataFrame(combi.columns)
variables.columns = ['column_name']
variables['min'] = 0
variables['max'] = 0
variables['distinct_count'] = 0
variables['sd'] = 0
variables['na_count'] = 0
variables['type'] = ''

for i in range(0,len(combi.columns)):
    variables.iloc[i,1] = combi.iloc[:,i].min()
    variables.iloc[i,2] = combi.iloc[:,i].max()
    variables.iloc[i,3] = len(combi.iloc[:,i].unique())
    variables.iloc[i,4] = np.std(combi.iloc[:,i])
    variables.iloc[i,5] = sum(pd.isnull(combi.iloc[:,i]))
    variables.iloc[i,6] = str(type(combi.iloc[0,i]))
    if i % 75 == 0:
        print(i)

"""
    - every variable is numeric. 
    - v1 variable --999999.00 value: I'll leave it here for now.
    
    - LAST COLUMN is the TARGET!!!
    - FIRST COLUMN is the ID!!!
"""
#Separate train and test tables from combi 
train = combi.iloc[0:76020,:]
test = combi.iloc[76020:,:]

#Write out current state: 
pd.DataFrame.to_csv(train, 'train_FE_v1.csv', index = False)
pd.DataFrame.to_csv(test, 'test_FE_v1.csv', index = False)

###############

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v1.csv")


##############################################################################
#               Check for highly correlated features
##############################################################################

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v1.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v1.csv")

train_corr = train.corr()
test_corr = test.corr()

'''
#Plotting correlation matrix - 
#   this is how we can plot a correlation matrix in python
# this is working, but have no point, because we have way too many variables

# Generate a mask for the upper triangle
mask = np.zeros_like(train_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(train_corr, mask=mask, cmap=cmap, vmax=.3,
            square=True, xticklabels=5, yticklabels=5,
            linewidths=.5, cbar_kws={"shrink": .5},)
'''

#Turn values below the diagonal into 0 for better comparison. 
for i in range(0, len(train_corr)):
    train_corr.iloc[i:,i] = 0
    test_corr.iloc[i:,i] = 0

pd.DataFrame.to_csv(train_corr, 'train_correlation.csv')
pd.DataFrame.to_csv(test_corr, 'test_correlation.csv')

resulttable = pd.DataFrame(train_corr.columns)
resulttable.columns = ['row_id']
resulttable['train_col_id'] = ''
resulttable['train_corr_value'] = 0
resulttable['test_col_id'] = ''
resulttable['test_corr_value'] = 0

for i in range(0, len(train_corr)):
    temp = train_corr.iloc[i,:]
    max_id = temp.idxmax()
    a = temp[max_id]
    resulttable.train_col_id.iloc[i] = max_id
    resulttable.train_corr_value.iloc[i] = a
    
for i in range(0, len(test_corr)):
    temp = test_corr.iloc[i,:]
    max_id = temp.idxmax()
    a = temp[max_id]
    resulttable.test_col_id.iloc[i] = max_id
    resulttable.test_corr_value.iloc[i] = a

'''
Correlation is 1 for the following correlation pairs: 
row_id train_col_id  train_corr_value test_col_id  test_corr_value
24     v26          v61                 1         v61         1.000000
50     v52          v54                 1         v54         1.000000
53     v55          v56                 1         v56         1.000000
62     v68          v69                 1         v69         1.000000
66     v72          v73                 1        v146         1.000000
69     v75          v76                 1         v76         1.000000
73     v79          v82                 1        v154         1.000000
83     v92         v136                 1        v136         1.000000
84     v93         v137                 1        v137         1.000000
107   v116         v117                 1        v117         1.000000
109   v118         v119                 1        v119         1.000000
129   v142         v143                 1        v143         1.000000
137   v150         v151                 1        v151         1.000000
141   v154         v157                 1        v157         1.000000
150   v166         v182                 1        v182         1.000000
155   v171         v351                 1        v351         0.979245
178   v198         v210                 1        v210         1.000000
179   v199         v211                 1        v211         1.000000
182   v202         v214                 1        v214         1.000000
184   v204         v216                 1        v216         1.000000
185   v205         v217                 1          v0         0.000000
186   v206         v218                 1        v218         1.000000
----------
I'll drop the following 22 columns, because of the redundancy: 
'v26', 'v52', 'v55', 'v68', 'v72', 'v75', 'v79', 'v92', 'v93', 'v116', 'v118', 'v142', 'v150', 'v154', 'v166', 'v171', 'v198', 'v199', 'v202', 'v204', 'v205', 'v206'

''' 
combi = pd.concat([train, test], axis = 0)

combi = combi.drop(['v26', 'v52', 'v55', 'v68', 'v72', 'v75', 'v79', 'v92', 'v93', 'v116', 'v118', 'v142', 'v150', 'v154', 'v166', 'v171', 'v198', 'v199', 'v202', 'v204', 'v205', 'v206'], axis = 1)

#Separate train and test tables from combi 
train = combi.iloc[0:76020,:]
test = combi.iloc[76020:,:]

#Write out current state: 
pd.DataFrame.to_csv(train, 'train_FE_v2.csv', index = False)
pd.DataFrame.to_csv(test, 'test_FE_v2.csv', index = False)


##########
#Next round of correlation check

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v2.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v2.csv")

train_corr = train.corr()
test_corr = test.corr()

for i in range(0, len(train_corr)):
    train_corr.iloc[i:,i] = 0
    test_corr.iloc[i:,i] = 0

resulttable = pd.DataFrame(train_corr.columns)
resulttable.columns = ['row_id']
resulttable['train_col_id'] = ''
resulttable['train_corr_value'] = 0
resulttable['test_col_id'] = ''
resulttable['test_corr_value'] = 0

for i in range(0, len(train_corr)):
    temp = train_corr.iloc[i,:]
    max_id = temp.idxmax()
    a = temp[max_id]
    resulttable.train_col_id.iloc[i] = max_id
    resulttable.train_corr_value.iloc[i] = a
    
for i in range(0, len(test_corr)):
    temp = test_corr.iloc[i,:]
    max_id = temp.idxmax()
    a = temp[max_id]
    resulttable.test_col_id.iloc[i] = max_id
    resulttable.test_corr_value.iloc[i] = a

resulttable.loc[resulttable.train_corr_value >0.99]
'''
By checking again, there are new variable pairs, where the correlation is 
very close to 1: 
----------
   row_id train_col_id  train_corr_value test_col_id  test_corr_value
24     v27         v137                 1        v137         1.000000
26     v29          v95                 1         v95         1.000000
34     v37         v103                 1         v38         1.000000
35     v38         v103                 1        v286         1.000000
41     v44         v110                 1        v195         1.000000
42     v45         v110                 1        v195         1.000000
44     v47         v112                 1        v112         1.000000
45     v48         v113                 1        v113         1.000000
52     v61         v136                 1        v136         1.000000
53     v62         v137                 1        v137         1.000000
62     v73         v146                 1        v146         1.000000
68     v82         v157                 1        v157         1.000000
70     v84         v161                 1        v161         0.993167
85    v103         v286                 1        v104         0.948680
86    v104         v286                 1        v286         1.000000
92    v110         v195                 1        v195         1.000000
93    v111         v195                 1        v195         1.000000
121   v146         v196                 1        v147         1.000000
122   v147         v196                 1        v196         1.000000
161   v197         v209                 1        v209         1.000000
162   v200         v212                 1        v212         1.000000
163   v201         v305                 1        v305         1.000000
164   v203         v309                 1          v0         0.000000
165   v207         v321                 1        v250         1.000000
166   v208         v220                 1        v220         1.000000
171   v213         v305                 1        v305         1.000000
173   v215         v309                 1          v0         0.000000
174   v216         v315                 1        v315         0.944906
175   v217         v317                 1          v0         0.000000
177   v219         v321                 1        v250         1.000000
192   v237         v306                 1          v0         0.000000
194   v240         v309                 1          v0         0.000000
202   v250         v321                 1        v321         1.000000
205   v253         v270                 1        v270         0.894425

The train_corr value is not 1, but it is higher than 0.9999, it is displayed
with roundings.
----------
I'll drop the following columns: 
'v27','v29','v37','v38','v44','v45','v47','v48','v61','v62','v73','v82','v84','v103','v104','v110','v111','v146','v147','v197','v200','v201','v203','v207','v208','v213','v215','v216','v217','v219','v237','v240','v250','v253'
'''
combi = pd.concat([train, test], axis = 0)

combi = combi.drop(['v27','v29','v37','v38','v44','v45','v47','v48','v61','v62','v73','v82','v84','v103','v104','v110','v111','v146','v147','v197','v200','v201','v203','v207','v208','v213','v215','v216','v217','v219','v237','v240','v250','v253'], axis = 1)

#Separate train and test tables from combi 
train = combi.iloc[0:76020,:]
test = combi.iloc[76020:,:]

#Write out current state: 
pd.DataFrame.to_csv(train, 'train_FE_v3.csv', index = False)
pd.DataFrame.to_csv(test, 'test_FE_v3.csv', index = False)

'''
After dropping these, checking the correlation once again, the max amount is
0.99986, which is still super high, but I decided to move forward with PCA. 

Another thing is, that submission trained on FE_v3 performs slightly worse. 
So, I might consider to revisit this last step, and stick with the v2...

So, in the next round I need to check correlation rates higher than .90/.95,
 and by using PCA, or something similar I might try to wrap up these variables. 
'''

##########
'''
ON FEv2 I need to drop highly correlated variables
First, I gather highly correlated variables. 
I build a model with both variables in, and excluding them one-by-one
If i get back better CV, I'll drop the one I excluded. 
'''
from sklearn import decomposition
from sklearn import preprocessing 

train = pd.read_csv("Documents/DataMining/Santander/Data/train_FE_v2.csv")
test = pd.read_csv("Documents/DataMining/Santander/Data/test_FE_v2.csv")
X = preprocessing.scale(train.iloc[:,:-1])

pca = decomposition.PCA(n_components = 314)
pca.fit(X = X , y = train.iloc[:,-1])
plt.plot(pca.explained_variance_ratio_)

#13 variables look like good by plotting explained cariance ratio. 
pca = decomposition.PCA(n_components = 182)
pca.fit(X = X, y = train.iloc[:,-1])
plt.plot(pca.explained_variance_ratio_)

#fit with applied dimensionality reduction
new_train = pca.fit_transform(X = X, y = train.iloc[:,-1])
new_train = pd.DataFrame(new_train)
test_X = preprocessing.scale(test.iloc[:,:-1])
new_test = pca.transform(X = test.iloc[:, :-1])
new_test = pd.DataFrame(new_test)

#Fit the model for new_train and new_test. 
submission = pd.read_csv("Documents/DataMining/Santander/CleanData/sample_submission.csv")

model = xgb.XGBClassifier(max_depth=5, n_estimators = 350, learning_rate=0.03, subsample = 0.95, colsample_bytree=0.85, silent = True, nthread = 4, seed = 4242)
X_fit, X_eval, y_fit, y_eval= train_test_split(new_train, train.iloc[:,-1], test_size=0.3)
model.fit(X = new_train, y = train.iloc[:,-1], early_stopping_rounds=20, eval_metric="auc", eval_set=[(X_eval, y_eval)])
prediction = model.predict_proba(new_test)

submission.TARGET = prediction
pd.DataFrame.to_csv(submission, 'xgb_v0_1FE2_fixedSeed_PCA.csv', index = False, )
#Leaderboard result:  0.472165 LOLOLOLOL.... This is a terrible result. 



