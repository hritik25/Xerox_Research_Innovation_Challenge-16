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
#from sklearn.cross_validation import train_test_split
 
 
 
df_labs_train = pd.read_csv('id_time_labs_train.csv')
df_vitals_train = pd.read_csv('id_time_vitals_train.csv')
df_age_train = pd.read_csv('id_age_train.csv')
 
df_variables_combined_train = df_labs_train.merge(df_vitals_train , on = ['ID','TIME'])    
df_variables_combined_train = df_variables_combined_train.drop(['L13', 'L17'], axis=1)  # Dropping the columns of L13 and L17
df_variables_combined_train['L20'] = df_variables_combined_train['L3']/df_variables_combined_train['L20']   #L20 now contains ratio of L3/L20
 
 
df_labs_val = pd.read_csv('id_time_labs_val.csv')
df_vitals_val = pd.read_csv('id_time_vitals_val.csv')
df_age_val = pd.read_csv('id_age_val.csv')
 
df_variables_combined_val = df_labs_val.merge(df_vitals_val , on = ['ID','TIME'])    
df_variables_combined_val = df_variables_combined_val.drop(['L13', 'L17'], axis=1)  # Dropping the columns of L13 and L17
df_variables_combined_val['L20'] = df_variables_combined_val['L3']/df_variables_combined_val['L20']   #L20 now contains ratio of L3/L20
 
 
 
#id_time_vitals = sys.argv[1]
#id_time_labs = sys.argv[2]
#id_age = sys.argv[3]
 
#df_labs_test = pd.read_csv(id_time_labs)
#df_vitals_test = pd.read_csv(id_time_vitals)
#df_age_test = pd.read_csv(id_age)
 
#df_variables_combined_test = df_labs_test.merge(df_vitals_test , on = ['ID','TIME'])    
#df_variables_combined_test = df_variables_combined_test.drop(['L13', 'L17'], axis=1)  # Dropping the columns of L13 and L17
#df_variables_combined_test['L20'] = df_variables_combined_test['L3']/df_variables_combined_test['L20']
# 
 
 
 
 
 
 
selected_features = ['L1_min', 'L2_std', 'L3_ratio_of_count', 'L4_last', 'L5_last', 'L6_min', 'L6_last', 'L7_last', 'L8_last', 'L9_min', 'L10_last', 'L11_last', 'L12_last', 'L14_exp_avg', 'L15_last', 'L15_mean', 'L18_exp_avg', 'L19_ratio_of_count', 'L20_ratio_of_count', 'L21_last', 'L23_max', 'L25_last', 'V1_min', 'V1_mean', 'V1_exp_avg', 'V1_last', 'V2_std', 'V2_ratio_of_count', 'V2_exp_avg', 'V2_last', 'V2_min', 'V3_last', 'V3_min', 'V3_exp_avg', 'V4_mean', 'V4_last', 'V4_std', 'V4_exp_avg', 'V4_min', 'V5_exp_avg', 'V5_last', 'V5_std', 'V5_mean', 'V6_last', 'V6_min', 'V6_exp_avg', 'AGE', 'L15_first_in_icu', 'L22_first_in_icu']
 
 
variables = df_variables_combined_train.columns.values.tolist()
variables.remove('ID')
variables.remove('ICU')
variables.remove('TIME')
 
suffixes = ['_last', '_exp_avg', '_mean', '_min', '_first_in_icu', '_std', '_ratio_of_count', '_max']
Lnames = ['L0']
Lnames = Lnames + variables[0:23]
Lnames.insert(13, 'L13')
Lnames.insert(17, 'L17')
Vnames = ['V0']
Vnames = Vnames + variables[-6:]
 
 
normal_values = { 'L1': 7.40 , 'L2': 40.0 , 'L3':87.5 , 'L4':140.0 , 'L5':4.3 ,'L6':25.0, 'L7':13.5, 'L8':0.89 , 'L9':7.25, 'L10':43.0 , 'L11':275.0 , 'L12':1.1 , 'L14':87.299555 , 'L15':1.35 , 'L16':0.05 , 'L18' : 125.0 ,'L19':85.0 , 'L20':4.5 , 'L21':4.4 , 'L22':95.5 , 'L23':39.546018 , 'L25':1.95,'V1':120.0 , 'V2':70.504731 , 'V3':83.006913 ,'V4':18, 'V5':97.5 , 'V6':98.2   }
 
 
 
 
def get_predicted_labels_and_median_time():
    output_df = pd.read_csv('output.csv')
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
 
    #3.for the last timestamp recorded for a patient
    #for this simply group the output_df dataframe by ID and take last()
    temp = output_df_grouped_by_id.last()
    temp = temp.drop('PREDICTION' , axis = 1)
    last_timestamp_df = temp.reset_index()
 
    #now calculate the mean prediction time 
    temp = first_prediction_of_death_df.merge(last_timestamp_df , on = ['ID'] , how  = 'inner' , suffixes =     ('_first_timestamp_of_one' , '_last_timestamp'))
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
        return -1
    elif sensitivity == 0.0:
        return -2
    elif median_prediction_time_clipped_at_72 < 5.0 :
        return -3
    else:
        median_prediction_time_score = float(median_prediction_time_clipped_at_72) / 72
        final_score = (0.75*sensitivity) + (0.2*median_prediction_time_score) + (0.05*specificity)
        print final_score        
        return final_score
 
 
 
def f_first_for_series(col):
    col = col[col.isnull() == False]
    if len(col) == 0:
        return None
    return col.iloc[0]
 
 
 
def f_exp_avg_for_series(col, alpha):
    exp_avg = None
    col = col[col.isnull() == False]
    flag = False
    for element in col:
        if not flag:
            exp_avg = element
            flag = True
        else:
            exp_avg = (alpha * element) + ( (1-alpha) * exp_avg )
    return exp_avg
 
def f_ratio_for_series(col):
    return float( col.count() )/col.shape[0]
 
 
def f_last_for_series(col):
    col = col[ col.isnull() == False ]
    col = col.values
    if len(col) == 0:
        return None
    return col[-1]
 
 
def fill_normal_in_missing(df):
    suffixes = ['_min' , '_max' , '_std' , '_last' , '_mean', '_exp_avg', '_first_in_icu']
 
    # dict of normal values
    #getting list of the variable names
    #variables_names = df_variables_combined_train.columns.values[2:31].tolist()
    for name in variables:
        for suffix in suffixes:
            if (name+suffix) in selected_features:
                if suffix == '_std':
                    df[name+suffix].fillna(0, inplace = True)
                else:
                    df[name+suffix].fillna(normal_values[name] , inplace = True)
    return df
 
 
 
feature_dict1 = {0: 'L1', 1: 'L2', 2: 'L3', 3: 'L4', 4: 'L5', 5: 'L6', 6: 'L6', 7: 'L7', 8: 'L8', 9: 'L9', 10: 'L10', 11: 'L11', 12: 'L12', 13: 'L14', 14: 'L15', 15: 'L15', 16: 'L18', 17: 'L19', 18: 'L20', 19: 'L21', 20: 'L23', 21: 'L25', 22: 'V1', 23: 'V1', 24: 'V1', 25: 'V1', 26: 'V2', 27: 'V2', 28: 'V2', 29: 'V2', 30: 'V2', 31: 'V3', 32: 'V3', 33: 'V3', 34: 'V4', 35: 'V4', 36: 'V4', 37: 'V4', 38: 'V4', 39: 'V5', 40: 'V5', 41: 'V5', 42: 'V5', 43: 'V6', 44: 'V6', 45: 'V6', 46: 'AGE', 47: 'L15', 48: 'L22'}            
 
for i in feature_dict1:
    feature_dict1[i] = feature_dict1[i].split('_')[0]
 
def fill_na_values(feature_list):
    normal_values = { 'L1': 7.40 , 'L2': 40.0 , 'L3':87.5 , 'L4':140.0 , 'L5':4.3 ,'L6':25.0, 'L7':13.5, 'L8':0.89 , 'L9':7.25, 'L10':43.0 , 'L11':275.0 , 'L12':1.1 , 'L14':87.299555 , 'L15':1.35 , 'L16':0.05 , 'L18' : 125.0 ,'L19':85.0 , 'L20':4.5 , 'L21':4.4 , 'L22':95.5 , 'L23':39.546018 , 'L25':1.95,'V1':120.0 , 'V2':70.504731 , 'V3':83.006913 ,'V4':18, 'V5':97.5 , 'V6':98.2   }
 
    for i in range(len(feature_list)):
 
        if pd.isnull( feature_list[i] ):
            feature_list[i] = normal_values[feature_dict1[i]]
 
    return feature_list
 
 
 
 
def get_patient_feature_values_val(patient_id, df_patient):
    #features = ['L1_min', 'L2_std', 'L3_ratio_of_count', 'L4_last', 'L5_last', 'L6_min', 'L6_last', 'L7_last', 'L8_last', 'L9_min', 'L10_last', 'L11_last', 'L12_last', 'L14_exp_avg', 'L15_last', 'L15_mean', 'L18_exp_avg', 'L19_ratio_of_count', 'L20_ratio_of_count', 'L21_last', 'L23_max', 'L25_last', 'V1_min', 'V1_mean', 'V1_exp_avg', 'V1_last', 'V2_std', 'V2_ratio_of_count', 'V2_exp_avg', 'V2_last', 'V2_min', 'V3_last', 'V3_min', 'V3_exp_avg', 'V4_mean', 'V4_last', 'V4_std', 'V4_exp_avg', 'V4_min', 'V5_exp_avg', 'V5_last', 'V5_std', 'V5_mean', 'V6_last', 'V6_min', 'V6_exp_avg']
    list_to_return = []
 
    list_to_return.append(df_patient['L1'].min())
    list_to_return.append(df_patient['L2'].std() )
    list_to_return.append(f_ratio_for_series( df_patient['L3'] ) )
    list_to_return.append(f_last_for_series( df_patient['L4'] ) )
    list_to_return.append(f_last_for_series( df_patient['L5'] ) )
    list_to_return.append(df_patient['L6'].min() )
    list_to_return.append(f_last_for_series( df_patient['L6'] ) )
    list_to_return.append(f_last_for_series( df_patient['L7'] ) )
    list_to_return.append(f_last_for_series( df_patient['L8'] ) )
    list_to_return.append(df_patient['L9'].min() )
    list_to_return.append(f_last_for_series( df_patient['L10'] ) )
    list_to_return.append(f_last_for_series( df_patient['L11'] ) )
    list_to_return.append(f_last_for_series( df_patient['L12'] ) )
    list_to_return.append(f_exp_avg_for_series( df_patient['L14'], 0.3) )
    list_to_return.append(f_last_for_series( df_patient['L15'] ) )
    list_to_return.append(df_patient['L15'].mean() )
    list_to_return.append(f_exp_avg_for_series( df_patient['L18'], 0.3 ) )
    list_to_return.append(f_ratio_for_series( df_patient['L19'] ) )
    list_to_return.append(f_ratio_for_series( df_patient['L20'] ) )
    list_to_return.append(f_last_for_series( df_patient['L21'] ) )
    list_to_return.append(df_patient['L23'].max() )
    list_to_return.append(f_last_for_series( df_patient['L25'] ) )
    list_to_return =  list_to_return + [ df_patient['V1'].min(), df_patient['V1'].mean(), f_exp_avg_for_series( df_patient['V1'], 0.3), f_last_for_series( df_patient['V1'])]
    list_to_return =  list_to_return + [ df_patient['V2'].std(), f_ratio_for_series( df_patient['V2']), f_exp_avg_for_series( df_patient['V2'], 0.3), f_last_for_series( df_patient['V2']), df_patient['V2'].min() ]
    list_to_return =  list_to_return + [ f_last_for_series( df_patient['V3']), df_patient['V3'].min(), f_exp_avg_for_series( df_patient['V3'], 0.3)]
    list_to_return =  list_to_return + [ df_patient['V4'].mean(), f_last_for_series( df_patient['V4']), df_patient['V4'].std(), f_exp_avg_for_series( df_patient['V4'], 0.3), df_patient['V4'].min()]
    list_to_return =  list_to_return + [ f_exp_avg_for_series( df_patient['V5'], 0.3), f_last_for_series( df_patient['V5']), df_patient['V5'].std(), df_patient['V5'].mean()]
    list_to_return =  list_to_return + [ f_last_for_series( df_patient['V6']), df_patient['V6'].min(), f_exp_avg_for_series( df_patient['V6'], 0.3)]
    age = df_age_val[ df_age_val['ID'] == patient_id ]['AGE'].iloc[0]
    list_to_return.append(age)
 
    final_list = fill_na_values(list_to_return)
 
    return final_list
 
 
 
#def get_patient_feature_values_test(patient_id, df_patient):
#    #features = ['L1_min', 'L2_std', 'L3_ratio_of_count', 'L4_last', 'L5_last', 'L6_min', 'L6_last', 'L7_last', 'L8_last', 'L9_min', 'L10_last', 'L11_last', 'L12_last', 'L14_exp_avg', 'L15_last', 'L15_mean', 'L18_exp_avg', 'L19_ratio_of_count', 'L20_ratio_of_count', 'L21_last', 'L23_max', 'L25_last', 'V1_min', 'V1_mean', 'V1_exp_avg', 'V1_last', 'V2_std', 'V2_ratio_of_count', 'V2_exp_avg', 'V2_last', 'V2_min', 'V3_last', 'V3_min', 'V3_exp_avg', 'V4_mean', 'V4_last', 'V4_std', 'V4_exp_avg', 'V4_min', 'V5_exp_avg', 'V5_last', 'V5_std', 'V5_mean', 'V6_last', 'V6_min', 'V6_exp_avg']
#    list_to_return = []
#    
#    list_to_return.append(df_patient['L1'].min())
#    list_to_return.append(df_patient['L2'].std() )
#    list_to_return.append(f_ratio_for_series( df_patient['L3'] ) )
#    list_to_return.append(f_last_for_series( df_patient['L4'] ) )
#    list_to_return.append(f_last_for_series( df_patient['L5'] ) )
#    list_to_return.append(df_patient['L6'].min() )
#    list_to_return.append(f_last_for_series( df_patient['L6'] ) )
#    list_to_return.append(f_last_for_series( df_patient['L7'] ) )
#    list_to_return.append(f_last_for_series( df_patient['L8'] ) )
#    list_to_return.append(df_patient['L9'].min() )
#    list_to_return.append(f_last_for_series( df_patient['L10'] ) )
#    list_to_return.append(f_last_for_series( df_patient['L11'] ) )
#    list_to_return.append(f_last_for_series( df_patient['L12'] ) )
#    list_to_return.append(f_exp_avg_for_series( df_patient['L14'], 0.3) )
#    list_to_return.append(f_last_for_series( df_patient['L15'] ) )
#    list_to_return.append(df_patient['L15'].mean() )
#    list_to_return.append(f_exp_avg_for_series( df_patient['L18'], 0.3 ) )
#    list_to_return.append(f_ratio_for_series( df_patient['L19'] ) )
#    list_to_return.append(f_ratio_for_series( df_patient['L20'] ) )
#    list_to_return.append(f_last_for_series( df_patient['L21'] ) )
#    list_to_return.append(df_patient['L23'].max() )
#    list_to_return.append(f_last_for_series( df_patient['L25'] ) )
#    list_to_return =  list_to_return + [ df_patient['V1'].min(), df_patient['V1'].mean(), f_exp_avg_for_series( df_patient['V1'], 0.3), f_last_for_series( df_patient['V1'])]
#    list_to_return =  list_to_return + [ df_patient['V2'].std(), f_ratio_for_series( df_patient['V2']), f_exp_avg_for_series( df_patient['V2'], 0.3), f_last_for_series( df_patient['V2']), df_patient['V2'].min() ]
#    list_to_return =  list_to_return + [ f_last_for_series( df_patient['V3']), df_patient['V3'].min(), f_exp_avg_for_series( df_patient['V3'], 0.3)]
#    list_to_return =  list_to_return + [ df_patient['V4'].mean(), f_last_for_series( df_patient['V4']), df_patient['V4'].std(), f_exp_avg_for_series( df_patient['V4'], 0.3), df_patient['V4'].min()]
#    list_to_return =  list_to_return + [ f_exp_avg_for_series( df_patient['V5'], 0.3), f_last_for_series( df_patient['V5']), df_patient['V5'].std(), df_patient['V5'].mean()]
#    list_to_return =  list_to_return + [ f_last_for_series( df_patient['V6']), df_patient['V6'].min(), f_exp_avg_for_series( df_patient['V6'], 0.3)]
#    age = df_age_test[ df_age_test['ID'] == patient_id ]['AGE'].iloc[0]
#    list_to_return.append(age)
#    
#    final_list = fill_na_values(list_to_return)
#    
#    return final_list
 
 
 
 
 
 
alpha = 0.3
 
#def feature_extraction_train():
#    """
#    will return a pd.Dataframe with 291 features extracted
#    from time series as columns and 2 columns more i.e. ID and age.
#    So total 293 columns
#    """
#    df_without_time_train = df_variables_combined_train.drop(['TIME'], axis=1)
#    df_grouped_by_id_train = df_without_time_train.groupby('ID')
#    
#    df_without_time_in_icu_only_train = df_without_time_train[df_without_time_train['ICU'] == 1]
#    df_grouped_by_id_in_icu_only_train = df_without_time_in_icu_only_train.groupby('ID')
#    L_15_group2 = df_grouped_by_id_in_icu_only_train['L15']
#    L_22_group2 = df_grouped_by_id_in_icu_only_train['L22']
#    
#    L1group = df_grouped_by_id_train['L1']
#    L2group = df_grouped_by_id_train['L2']
#    L3group = df_grouped_by_id_train['L3']
#    L4group = df_grouped_by_id_train['L4']
#    L5group = df_grouped_by_id_train['L5']
#    L6group = df_grouped_by_id_train['L6']
#    L7group = df_grouped_by_id_train['L7']
#    L8group = df_grouped_by_id_train['L8']
#    L9group = df_grouped_by_id_train['L9']
#    L10group = df_grouped_by_id_train['L10']
#    L11group = df_grouped_by_id_train['L11']
#    L12group = df_grouped_by_id_train['L12']
#    L14group = df_grouped_by_id_train['L14']
#    L15group = df_grouped_by_id_train['L15']
#    #L16group = df_grouped_by_id_train['L16']
#    L18group = df_grouped_by_id_train['L18']
#    L19group = df_grouped_by_id_train['L19']
#    L20group = df_grouped_by_id_train['L20']
#    L21group = df_grouped_by_id_train['L21']
#    #L22group = df_grouped_by_id_train['L22']
#    L23group = df_grouped_by_id_train['L23']
#    #L24group = df_grouped_by_id_train['L24']
#    L25group = df_grouped_by_id_train['L25']
#    V1group = df_grouped_by_id_train['V1']
#    V2group = df_grouped_by_id_train['V2']
#    V3group = df_grouped_by_id_train['V3']
#    V4group = df_grouped_by_id_train['V4']
#    V5group = df_grouped_by_id_train['V5']
#    V6group = df_grouped_by_id_train['V6']
#    
#    
#    
#    df_L1 = L1group.agg( {Lnames[1]+suffixes[3] : np.min} ).reset_index()
#    df_L2 = L2group.agg( {Lnames[2]+suffixes[5] : np.std } ).reset_index()
#    x = pd.merge( df_L1, df_L2, on = 'ID')
#    
#    
#    df_L3 = L3group.agg( { Lnames[3]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L3, on = 'ID')
#    
#    df_L4 = L4group.agg( { Lnames[4]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L4, on = 'ID')
#    
#    df_L5 = L5group.agg( { Lnames[5]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L5, on = 'ID')
#    
#    df_L6 = L6group.agg( { Lnames[6]+suffixes[0] : lambda x: f_last_for_series(x), Lnames[6]+suffixes[3] : np.min } ).reset_index()
#    x = pd.merge( x, df_L6, on = 'ID')
#    
#    df_L7 = L7group.agg( { Lnames[7]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L7, on = 'ID')
#    
#    df_L8 = L8group.agg( { Lnames[8]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L8, on = 'ID')
#    
#    df_L9 = L9group.agg( { Lnames[9]+suffixes[3] : np.min } ).reset_index()
#    x = pd.merge( x, df_L9, on = 'ID')
#    
#    df_L10 = L10group.agg( { Lnames[10]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L10, on = 'ID')
#    
#    df_L11 = L11group.agg( { Lnames[11]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L11, on = 'ID')
#    
#    df_L12 = L12group.agg( { Lnames[12]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L12, on = 'ID')
#    
#    df_L14 = L14group.agg( { Lnames[14]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) } ).reset_index()
#    x = pd.merge( x, df_L14, on = 'ID')
#    
#    df_L15 = L15group.agg( { Lnames[15]+suffixes[0] : lambda x: f_last_for_series(x), Lnames[15]+suffixes[2] : np.mean } ).reset_index()
#    x = pd.merge( x, df_L15, on = 'ID')
#    
#    df_L18 = L18group.agg( { Lnames[18]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) } ).reset_index()
#    x = pd.merge( x, df_L18, on = 'ID')
#    
#    df_L19 = L19group.agg( { Lnames[19]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L19, on = 'ID')
#    
#    df_L20 = L20group.agg( { Lnames[20]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L20, on = 'ID')
#    
#    df_L21 = L21group.agg( { Lnames[21]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L21, on = 'ID')
#    
#    df_L23 = L23group.agg( { Lnames[23]+suffixes[7] : np.max } ).reset_index()
#    x = pd.merge( x, df_L23, on = 'ID')
#    
#    df_L25 = L25group.agg( { Lnames[25]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
#    x = pd.merge( x, df_L25, on = 'ID')
#    
#    df_V1 = V1group.agg( { Vnames[1]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[1]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[1]+suffixes[2] : np.mean, Vnames[1]+suffixes[3] : np.min} ).reset_index()
#    x = pd.merge( x, df_V1, on = 'ID')
#    
#    df_V2 = V2group.agg( { Vnames[2]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[2]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[2]+suffixes[3] : np.min, Vnames[2]+suffixes[5] : np.std, Vnames[2]+suffixes[6] : lambda x: f_ratio_for_series(x)} ).reset_index()
#    x = pd.merge( x, df_V2, on = 'ID')
#    
#    df_V3 = V3group.agg( { Vnames[3]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[3]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[3]+suffixes[3] : np.min} ).reset_index()
#    x = pd.merge( x, df_V3, on = 'ID')
#    
#    df_V4 = V4group.agg( { Vnames[4]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[4]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[4]+suffixes[2] : np.mean, Vnames[4]+suffixes[3] : np.min, Vnames[4]+suffixes[5] : np.std} ).reset_index()
#    x = pd.merge( x, df_V4, on = 'ID')
#    
#    df_V5 = V5group.agg( { Vnames[5]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[5]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[5]+suffixes[2] : np.mean, Vnames[5]+suffixes[5] : np.std} ).reset_index()
#    x = pd.merge( x, df_V5, on = 'ID')
#    
#    df_V6 = V6group.agg( { Vnames[6]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[6]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[6]+suffixes[3] : np.min} ).reset_index()
#    x = pd.merge( x, df_V6, on = 'ID')
#    
#    
#    
#    x = pd.merge( x, df_age_train, on = 'ID')  # appending age
#    
#    df_L15_2 = L_15_group2.agg( {'L15_first_in_icu' : lambda x: f_first_for_series(x)} ).reset_index()
#    x = pd.merge( x, df_L15_2, on = 'ID')
#    
#    df_L22_2 = L_22_group2.agg( {'L22_first_in_icu' : lambda x: f_first_for_series(x)} ).reset_index()
#    x = pd.merge( x, df_L22_2, on = 'ID')
#    
#    
#    #finally the labels
#    df_labels_train = pd.read_csv('id_label_train.csv')
#    x = pd.merge(x, df_labels_train , on = 'ID') 
#      
#    #print x.columns.values
#    #so y is our final dataframe which will be passed to the featue selection algorithms
#    y  =  fill_normal_in_missing(x)
#    #print y[y.isnull() == True]
#    return y
 
starting_time = time.time()
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
 
"""
gbc = GradientBoostingClassifier()
gbc.fit(X_train, Y_train)
"""
clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(min_samples_leaf =2) , n_estimators=150)   
clf.fit(X_train , Y_train)
 
fitting_time = time.time()
 
#rf = RandomForestClassifier(random_state=0)
 
 
#f = RandomForestClassifier(random_state=0).fit(X_train, Y_train)
#y_prob_train = f.predict_proba(X_train)
#y_prob_test = f.predict_proba(X_val)
#y_pred_test_rf = f.predict(X_val)
#f_t = ThresholdingOptimization().fit(y_prob_train, cost_mat_train, Y_train)
#y_pred_test_rf_t = f_t.predict(y_prob_test)
 
#c.fit(X_train, Y_train, cost_mat_train)
#y_pred_test_c = c.predict(X_val, cost_mat_val)
#y_prob_test_c = c.predict_proba(X_val)
 
 
#rf.fit(X_train, Y_train)
#y_pred_test_rf = rf.predict(X_val)
#y_prob_test_rf = rf.predict_proba(X_val)
##y_prob_train_rf = rf.predict_proba(X_train)
 
#predicted_y_c = y_pred_test_c.tolist()
#predicted_y_c = y_pred_test_c.tolist()
#predicted_y_rf = y_pred_test_rf.tolist()
#predicted_y_csrf = y_pred_test_csrf.tolist()
 
 
 
"""
training ends here
"""
 
 
 
 
 
"""
Validation starts here
"""
 
 
output = []
 
min_id_val = df_age_val['ID'].min()
max_id_val = df_age_val['ID'].max()
 
grouped_by_id_val = df_variables_combined_val.groupby('ID')
 
for i in range(min_id_val , max_id_val + 1):
 
    current_patient_df_val = grouped_by_id_val.get_group(i)
    counter = 0
    for j,row in current_patient_df_val.iterrows():
        flag1 = 0
        l15_first_in_icu_val = normal_values['L15']
        flag2 = 0   
        l22_first_in_icu_val = normal_values['L22']
        if row[-1] == 1:
            if (not flag1) & ( not math.isnan(row[15]) ):
                l15_first_in_icu_val = row[15]
                flag1 = 1
            if (not flag2) & ( not math.isnan(row[21]) ):
                l22_first_in_icu_val = row[21]
                flag2 = 1
            data_so_far_df_val = current_patient_df_val.head(counter + 1)
            #pass this data to the function which will do stuff with this and return a np.array which is supposed to be the 
        #input feature vector for our classifier model
 
            feature_vector_val = get_patient_feature_values_val(i, data_so_far_df_val.drop(['ID','TIME'],axis = 1))
 
            feature_vector_val.append(l15_first_in_icu_val)
            feature_vector_val.append(l22_first_in_icu_val)
 
            feature_vector_val = np.array(feature_vector_val)
            feature_vector_val = np.array( [feature_vector_val] )
 
 
            y_predicted_val = clf.predict(feature_vector_val)[0]
 
 
            """
            Space left for the classification code of prediciton .
            At the end of this code , we will have a value as 0 or 1 which is the PREDICTION for this patient for this Timestamp
            Let's have it returned to the variable 'predicted'
            """
 
            output.append([row[0] , row[1] , y_predicted_val])
            counter = counter + 1
 
#now write the output to a csv file
wr  = csv.writer(open('output.csv' , 'wb'))
""" this is for the header , just for your own evaluation , but DON'T include this for Submission """
wr.writerow(['ID' , 'TIME' , 'PREDICTION'])
wr.writerows(output)
 
final_score = evaluate_score()
 
print final_score
 
 
 
 
 
 
 
"""
Validation ends here
"""
 
 
 
 
"""
Testing starts here
"""
 
#output_test = []
#
#min_id_test = df_age_test['ID'].min()
#max_id_test = df_age_test['ID'].max()
#
#grouped_by_id_test = df_variables_combined_test.groupby('ID')
#
#for i in range(min_id_test , max_id_test):
#            
#    current_patient_df_test = grouped_by_id_test.get_group(i)
#    for j,row in current_patient_df_test.iterrows():
#        #lets assume the index of ICU in the dataframe we have is 'index'
#     #start predicting if the patient goes to ICU
#     """Here DON'T forget to append L15 , L22 to
#        the last of the np.array you will be passing to the model, 
#        just do np.append """                
#        
#        flag = 0
#        if row[0] == 1:                 # here in row[0], 0 is used because the column number of ICU in current_patient_df is 0  
#            if not flag:
#                l15_first_in_icu_test = row[15]
#                l22_first_in_icu_test = row[21]
#                flag = 1
#            data_so_far_df_test = current_patient_df_test.head(j+1)
#            #pass this data to the function which will do stuff with this and return a np.array which is supposed to be the 
#       #input feature vector for our classifier model
#            
#            feature_vector_test = get_patient_feature_values(i, data_so_far_df_test.drop(['ID','TIME'],axis = 1))
#            
#            feature_vector_test.append(l15_first_in_icu_test)
#            feature_vector_test.append(l22_first_in_icu_test)
#            
#            feature_vector_test = np.array(feature_vector_test)
#            feature_vector_test = np.array( [feature_vector_test] )
#            
#            
#            y_predicted_test = c.predict(feature_vector_test)[0]
#            
#            """
#            Space left for the classifierode of prediciton .
#            At the end of this code , we will have a value as 0 or 1 which is the PREDICTION for this patient for this Timestamp
#            Let's have it returned to the variable 'predicted'
#            """
#            
#            output_test.append([row[0] , row[1] , y_predicted_test])
#
##now write the output to a csv file
#wr  = csv.writer(open('output_test.csv' , 'wb'))
#""" this is for the header , just for your own evaluation , but DON'T include this for Submission """
##wr.writerows(['ID' , 'TIME' , 'PREDICTION'])
#wr.writerows(output_test)
#
#
#"""
#Testing ends here
#"""