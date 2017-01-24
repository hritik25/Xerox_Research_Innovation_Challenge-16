import time
import csv
import sys
import math
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFpr
import random
import collections
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier , RandomForestClassifier, GradientBoostingClassifier
#from costcla.models import CostSensitiveRandomForestClassifier
 
 
selected_features = ['L1_min', 'L2_std', 'L3_ratio_of_count', 'L4_last', 'L5_last', 'L6_min', 'L6_last', 'L7_last', 'L8_last', 'L9_min', 'L10_last', 'L11_last', 'L12_last', 'L14_exp_avg', 'L15_last', 'L15_mean', 'L18_exp_avg', 'L19_ratio_of_count', 'L20_ratio_of_count', 'L21_last', 'L23_max', 'L25_last', 'V1_min', 'V1_mean', 'V1_exp_avg', 'V1_last', 'V2_std', 'V2_ratio_of_count', 'V2_exp_avg', 'V2_last', 'V2_min', 'V3_last', 'V3_min', 'V3_exp_avg', 'V4_mean', 'V4_last', 'V4_std', 'V4_exp_avg', 'V4_min', 'V5_exp_avg', 'V5_last', 'V5_std', 'V5_mean', 'V6_last', 'V6_min', 'V6_exp_avg', 'AGE', 'L15_first_in_icu', 'L22_first_in_icu']
 
 
 
def get_predicted_labels_and_median_time():
    output_df = pd.read_csv('output_gbc.csv')
    output_df_grouped_by_id = output_df.groupby('ID')
 
	#this is the dataframe with our prediction labels for patients 
    temp = output_df_grouped_by_id.max()
    temp = temp.reset_index()
    temp = temp.drop('TIME', axis = 1)
    temp.columns = ['ID' , 'LABEL']
    predicted_labels_df = temp
 
    #the list of predicted labels to return
    predicted_labels = np.array(predicted_labels_df['LABEL']).tolist()
 
 
 
    #2.for the first time you predicted death for a patient
    #for this first select only those rows which have timestamp predictions = 1
    temp = output_df[output_df['PREDICTION'] == 1]
    # group this temp dataframe by ID and drop the TIME stamp column , as this dataframe contains the first timestamp           where a patient's death was predicted , if it was
    # then take the first()
    temp = temp.groupby('ID').first()
    temp = temp.drop('PREDICTION' , axis = 1)
    first_prediction_of_death_df =  temp.reset_index()
    #print 'Positives ',first_prediction_of_death_df.shape , first_prediction_of_death_df.info()

    #for the new criteria uploaded on hackerrank , median prediction time only for True Positives
    actual_labels_df = pd.read_csv(open('id_label_val.csv'))
    actual_positives_df = actual_labels_df[actual_labels_df['LABEL'] == 1]
    temp = actual_positives_df.merge(first_prediction_of_death_df , on = ['ID'] , how  = 'inner')
    temp = temp.drop('LABEL' , axis = 1)
    first_prediction_of_death_only_for_true_positives_df = temp
    #print 'True Positives ', first_prediction_of_death_only_for_true_positives_df.shape , first_prediction_of_death_only_for_true_positives_df.info()

    #3.for the last timestamp recorded for a patient
    #for this simply group the output_df dataframe by ID and take last()
    temp = output_df_grouped_by_id.last()
    temp = temp.drop('PREDICTION' , axis = 1)
    last_timestamp_df = temp.reset_index()
 
    #now calculate the mean prediction time 
    temp = first_prediction_of_death_only_for_true_positives_df.merge(last_timestamp_df , on = ['ID'] , how  = 'inner' , suffixes = ('_first_timestamp_of_one' , '_last_timestamp'))
    temp['PREDICTION_TIME'] = temp['TIME_last_timestamp'] - temp['TIME_first_timestamp_of_one']
    median_prediction_time = temp['PREDICTION_TIME'].median()
    print 'median_prediction_time ' + str(median_prediction_time)
    return predicted_labels , median_prediction_time
 
 
 
"""
given two lists , one with predicted labels , one with actual labels
calculate calculate the no. of TNs , TPs , FPs , FNs and hence the specificity and sensitivity of predictions
"""
 
def get_results():
    #the four parameters of evaluation
    actual_labels_df = pd.read_csv(open('id_label_val.csv')) #filename hardcoded
    actual = np.array(actual_labels_df['LABEL']).tolist()
    predicted , median_prediction_time = get_predicted_labels_and_median_time()
 
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for i in range(len(predicted)):
        if (predicted[i] == 1) & (actual[i] == 1):
            tp = tp + 1
        elif (predicted[i] == 1) & (actual[i] == 0):
            fp = fp + 1
        elif (predicted[i] == 0) & (actual[i] == 0):
            tn = tn + 1
        else:
            fn = fn + 1
 
    """
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    """
 
    specificity = float(tn) / (tn + fp)
    sensitivity = float(tp)/ (tp + fn)
    # just noticed , accuracy not required
    # accuracy = float(tp + tn) / (tp + tn + fp + fn)
    print 'spec ' + str(specificity)
    print 'sens ' + str(sensitivity)
    print 'tp ', tp
    print 'tn ', tn
    print 'fp ', fp
    print 'fn ', fn
    return specificity , sensitivity , median_prediction_time, fp, fn, tp, tn
 
def evaluate_score():
    #get the following values

    specificity , sensitivity , median_prediction_time, fp , fn , tp , tn = get_results()
 
    median_prediction_time = float(median_prediction_time)/3600
 
    if median_prediction_time < 72.0:
        median_prediction_time_clipped_at_72 = median_prediction_time
    else:
        median_prediction_time_clipped_at_72 = 72 
 
    if specificity < 0.99:
        print 'Score will be 0 because specificity is < 0.99'
        return -1
    elif sensitivity == 0.0:
        print 'Score will be 0 because sensitivity is 0'
        return -2
    elif median_prediction_time_clipped_at_72 < 5.0 :
        print 'Score will be 0 because median prediction time is < 5 hours'
        return -3
    else:
        median_prediction_time_score = float(median_prediction_time_clipped_at_72) / 72
        final_score = (0.75*sensitivity) + (0.2*median_prediction_time_score) + (0.05*specificity)
        print 'Final Score' , final_score        
        return final_score
 
 
 
starting_point = time.time()
"""
training starts here
"""
#df_train = feature_extraction_train()  
df_train = pd.read_csv('buffer_train.csv')
df_X_train = df_train.drop(['ID', 'LABEL', 'Unnamed: 0'], axis=1)
#df_X_train = df_train.drop(['ID', 'LABEL'], axis=1)
X_train = df_X_train.values
#X_train[X_train.columns.values.tolist()] = X_train[X_train.columns.values.tolist()].astype(np.float32)
Y_train = df_train[ 'LABEL' ].values


 

just_before_fitting_point = time.time()
delta_time = just_before_fitting_point - starting_point
 

id_time_features_df = pd.read_csv('id_time_features.csv')
X_val = id_time_features_df.drop(['ID','TIME'],axis = 1)


# print 'starting fitting at ' + str(delta_time) + ' time after start\n\n'
# for ne in [50 , 75 , 100 , 125 ,150 ,175 ,200]:
#     print  '\n\n\n\nNo.  of estimators =' , ne
#     for leaf in range(1,4):
#         print '\n\n\nmin_samples_leaf =' ,leaf
#         for m_samples_split in range(2,5):
#             print '\nmin_samples_split =' , m_samples_split
#             gbc=GradientBoostingClassifier() 
#             gbc.fit(X_train , Y_train)
#             # for maxdepth in range(3,4):
#             #     for i in [100]:
#             #         gbc = GradientBoostingClassifier( n_estimators = 375 , max_depth = 3 , min_samples_leaf = 5)
#             #         gbc.fit(X_train , Y_train)
                    

#             fitting_point = time.time()
#             fitting_time = fitting_point - starting_point
             
#             print 'fit time ', fitting_time
             
#             """
#             training ends here
#             """
#             """
#             validation starts here
#             """
             
#             #print X_val.info()

#             predicted_labels = clf.predict(X_val)
#             predicted_probabilities = clf.predict_proba(X_val)
#             #print 'predicted probabilities ', predicted_probabilities
#             #print 'probabilities which give Prediction 1\n' 
#             predicted_labels = []
#             for i in range(len(predicted_probabilities)):
#                 if predicted_probabilities[i][1] > 0.999:
#                     predicted_labels.append(1)
#                 else:
#                     predicted_labels.append(0)

#             # predicted_probabilities = gbc.predict_proba(X_val)
#             # print 'predicted probabilities ', predicted_probabilities
#             # #print 'probabilities which give Prediction 1\n' 
#             # predicted_labels = []
#             # for i in range(len(predicted_probabilities)):
#             #     if predicted_probabilities[i][1] > 0.99995:
#             #         predicted_labels.append(1)
#             #     else:
#             #         predicted_labels.append(0)





#             output_df = id_time_features_df[['ID' , 'TIME']]
#             #print output_df.info()

#             output_df['PREDICTION'] = pd.Series(predicted_labels)
#             #print output_df
             
             
#             wr  = csv.writer(open('output_gbc.csv' , 'w+'))
#             wr.writerow(['ID' , 'TIME' , 'PREDICTION'])
#             wr.writerows(output_df.values)

#             final_score = evaluate_score()

rfc=RandomForestClassifier() 
rfc.fit(X_train , Y_train)
            # for maxdepth in range(3,4):
            #     for i in [100]:
            #         gbc = GradientBoostingClassifier( n_estimators = 375 , max_depth = 3 , min_samples_leaf = 5)
            #         gbc.fit(X_train , Y_train)
                    

fitting_point = time.time()
fitting_time = fitting_point - starting_point
             
print 'fit time ', fitting_time
             
             
            #print X_val.info()

predicted_labels_rfc = rfc.predict(X_val)
predicted_probabilities = rfc.predict_proba(X_val)
            #print 'predicted probabilities ', predicted_probabilities
            #print 'probabilities which give Prediction 1\n' 
# for i in range(len(predicted_probabilities)):
#     if predicted_labels_rfc[i] == 1:
#         print predicted_probabilities[i]
predicted_labels = []
for i in range(len(predicted_probabilities)):
    if predicted_probabilities[i][1] > 0.99:
        predicted_labels.append(1)
    else:
        predicted_labels.append(0)

            # predicted_probabilities = gbc.predict_proba(X_val)
            # print 'predicted probabilities ', predicted_probabilities
            # #print 'probabilities which give Prediction 1\n' 
            # predicted_labels = []
            # for i in range(len(predicted_probabilities)):
            #     if predicted_probabilities[i][1] > 0.99995:
            #         predicted_labels.append(1)
            #     else:
            #         predicted_labels.append(0)





output_df = id_time_features_df[['ID' , 'TIME']]
            #print output_df.info()

output_df['PREDICTION'] = pd.Series(predicted_labels_rfc)
            #print output_df
             
             
wr  = csv.writer(open('output_gbc.csv' , 'w+'))
wr.writerow(['ID' , 'TIME' , 'PREDICTION'])
wr.writerows(output_df.values)

final_score = evaluate_score()

