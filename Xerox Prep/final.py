import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFpr
import random
import collections
from sklearn.ensemble import RandomForestClassifier
# dict of normal values
normal_values = { 'L1': 7.40 , 'L2': 40.0 , 'L3':87.5 , 'L4':140.0 , 'L5':4.3 ,'L6':25.0, 'L7':13.5, 'L8':0.89 , 'L9':7.25, 'L10':43.0 , 'L11':275.0 , 'L12':1.1 , 'L14':87.299555 , 'L15':1.35 , 'L16':0.05 , 'L18' : 125.0 ,'L19':85.0 , 'L20':4.5 , 'L21':4.4 , 'L22':95.5 , 'L23':39.546018,'L24':45 , 'L25':1.95,'V1':120.0 , 'V2':70.504731 , 'V3':83.006913 ,'V4':18, 'V5':97.5 , 'V6':98.2	}
 
df_labs = pd.read_csv('id_time_labs_train.csv')
df_vitals = pd.read_csv('id_time_vitals_train.csv')
df_age = pd.read_csv('id_age_train.csv')
 
df_variables_combined = df_labs.merge(df_vitals , on = ['ID','TIME'])
 
df_variables_combined = df_variables_combined.drop(['L13', 'L17'], axis=1)  # Dropping the columns of L13 and L17
 
df_variables_combined['L20'] = df_variables_combined['L3']/df_variables_combined['L20']   #L20 now contains ratio of L3/L20

#getting list of the variable names
variables_names = df_variables_combined.columns.values[2:31].tolist()
#variables_names.remove('V4')

def fill_normal_in_missing(df):
	suffixes = ['_min' , '_max' , '_std' , '_last' , '_mean'	]
	
	for name in variables_names:	
		for suffix in suffixes:
			if suffix == '_std' or suffix == '_trend':
				df[name+suffix].fillna(0, inplace = True)
			else:
				df[name+suffix].fillna(normal_values[name] , inplace = True)
	return df
 
#print df_variables_combined.columns.values  
#print df_variables_combined
 
df_without_time = df_variables_combined.drop(['TIME'], axis=1)
  
df_grouped_by_id = df_without_time.groupby('ID')
 
min = df_grouped_by_id.min().reset_index()
max = df_grouped_by_id.max().reset_index()
 
x = pd.merge( min, max, on = 'ID', suffixes = ('_min','_max') ) # joining min and max
 
x = pd.merge( df_age, x, on = 'ID')  # appending age
 
std = df_grouped_by_id.std().reset_index()
x = pd.merge( x, std, on = 'ID' )  # appending std
 
last = df_grouped_by_id.last().reset_index()
x = pd.merge( x, last, on = 'ID', suffixes = ('_std', '_last' ) )  #appending last
 
mean = df_grouped_by_id.mean().reset_index()
x = pd.merge( x, mean, on = 'ID')  # appending mean

def f_ratio(y):
    return y.count()/y.shape[0]
 
ratio_of_values_measured = df_grouped_by_id.apply( f_ratio ).reset_index() #lambda y : y.count()/y.shape[0] )
x = pd.merge( x, ratio_of_values_measured, on = 'ID', suffixes = ('_mean', '_ratio_of_count' ) )  #appending ratio_of_count

#finally the labels
labels = pd.read_csv('id_label_train.csv')
x = pd.merge(x, labels , on = 'ID') 

x = x.drop(['ICU_mean', 'ICU_min', 'ICU_std', 'ICU_last', 'ICU_max', 'ICU_ratio_of_count'], axis=1) 

#print x.columns.values
#so y is our final dataframe which will be passed to the featue selection algorithms
y  =  fill_normal_in_missing(x)



#$updated after transferring code

data_train_df= y.drop('ID' , axis = 1)
data_train_df = data_train_df.drop('LABEL' , axis = 1)
target_train = y['LABEL'].values
names = data_train_df.columns.values
data_train = data_train.values

#feature selection with SelectFPR
s = SelectFpr()
data_train_filtered = s.fit_transform(data_train , target_train)
fpr_selected = []
selected_features_fpr = s.get_support(indices = True)
for i in range(len(selected_features_fpr)):
	fpr_selected.append(names[selected_features_fpr[i]])
 
#$now fpr_selected are the new selected variables
#$futher feature selection algorithms should map their top feature names to 'fpr_selected' and 'not names'

#applying Recursive Feature Elimination for feature selection
"""
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
sv = LinearSVC()
selector = RFE(sv)
selector.fit(data_train , target_train)
selected_features_svc = selector.get_support(indices=True)
ranks_rfe= selector.ranking_
top_names_rfe = []
for i in range(len(ranks)):
	if ranks[i] == 1:
		top_names_rfe.append(names[i])

"""
#Recursive Feature Elimination for Cross Validation for feature selection
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
sv = LinearSVC()
selector = RFECV(sv)
selector.fit(data_train_filtered , target_train)
ranks_rfecv = selector.ranking_
top_names_rfecv = []
for i in range(len(ranks_rfecv)):
	if ranks_rfecv[i] == 1:
		top_names_rfecv.append(fpr_selected[i])


#manual Random Forest Classifier Feature Selection

positive_samples = y[y['LABEL'] == 1].drop('ID' , axis =1)
negative_samples = y[y['LABEL'] == 0].drop('ID' , axis = 1)

top_features = np.array([])

flag = 0
for i in range(10):
	a =  positive_samples.ix[random.sample(positive_samples.index , 50)]
	b = negative_samples.ix[random.sample(negative_samples.index , 720)]
	test = pd.concat([a,b])
	test = test.iloc[np.random.permutation(len(test))]
	test_target = test['LABEL'].values
	test_train = test.drop('LABEL' , axis = 1)
#  project on the features selected from SelectFpr
	test_train = test_train[fpr_selected].values	
	rc= RandomForestClassifier()
	rc.fit(test_train , test_target)
	top_features_current = rc.feature_importances_
	if flag:
		top_features = top_features+top_features_current
	if not flag:
		top_features = top_features_current
		flag = 1

scores = {}
for i in range(len(top_features)):
	scores[top_features[i]] = names[i]

scores = collections.OrderedDict(sorted(scores.items() , reverse=True))

#rankwise top 40 by RFECV
ranks = []
for value , name in scores.iteritems():
	ranks.append(name)
top_forty = ranks[0:40]