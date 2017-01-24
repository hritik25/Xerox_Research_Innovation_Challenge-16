# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 07:05:50 2015
 
@author: ayush
"""
 
import pandas as pd
import numpy as np
import csv
from sklearn.feature_selection import SelectFpr
import random
import collections
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
# from costcla.datasets import load_creditscoring1
# from costcla.models import CostSensitiveRandomForestClassifier
# from costcla.metrics import savings_score
 
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
 
 
selected_features = ['V1_last', 'V1_exp_avg', 'V5_last', 'V5_exp_avg', 'V2_last', 'V3_last', 'V2_exp_avg', 'V4_last', 'V4_exp_avg', 'V6_last', 'L4_last', 'V6_exp_avg', 'L6_min', 'L15_mean', 'L21_last', 'L1_min', 'V2_min', 'V1_min', 'V1_mean', 'V6_min', 'V5_std', 'V4_std', 'V3_min', 'L15_last', 'V3_exp_avg', 'V2_ratio_of_count', 'L5_last', 'L10_last', 'L6_last', 'V2_std', 'L7_last', 'V4_mean', 'V5_mean', 'L8_last', 'V4_min', 'L15_first_in_icu', 'L12_last', 'L3_ratio_of_count', 'L2_std', 'L9_min', 'L11_last', 'L14_exp_avg', 'L18_exp_avg', 'L19_ratio_of_count', 'L20_ratio_of_count', 'L22_first_in_icu', 'L23_max', 'L25_last', 'AGE']
 
 
 
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
 
 
 
def get_results(predicted , actual):
    """
    given two lists , one with predicted labels , one with actual labels
    calculate calculate the no. of TNs , TPs , FPs , FNs and hence the specificity and sensitivity of predictions
    """
    #the four parameters of evaluation
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
    accuracy = float(tp + tn) / (tp + tn + fp + fn)
    return specificity, sensitivity, fp, fn, tp, tn
 
 
 
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
    suffixes = ['_min' , '_max' , '_std' , '_last' , '_mean', '_exp_avg']
 
    # dict of normal values
    normal_values = { 'L1': 7.40 , 'L2': 40.0 , 'L3':87.5 , 'L4':140.0 , 'L5':4.3 ,'L6':25.0, 'L7':13.5, 'L8':0.89 , 'L9':7.25, 'L10':43.0 , 'L11':275.0 , 'L12':1.1 , 'L14':87.299555 , 'L15':1.35 , 'L16':0.05 , 'L18' : 125.0 ,'L19':85.0 , 'L20':4.5 , 'L21':4.4 , 'L22':95.5 , 'L23':39.546018 , 'L25':1.95,'V1':120.0 , 'V2':70.504731 , 'V3':83.006913 ,'V4':18, 'V5':97.5 , 'V6':98.2	}
 
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
 
 
 
 
alpha = 0.3
 
def feature_extraction_train():
    """
    will return a pd.Dataframe with 291 features extracted
    from time series as columns and 2 columns more i.e. ID and age.
    So total 293 columns
    """
    df_without_time_train = df_variables_combined_train.drop(['TIME'], axis=1)
    df_grouped_by_id_train = df_without_time_train.groupby('ID')
 
    L1group = df_grouped_by_id_train['L1']
    L2group = df_grouped_by_id_train['L2']
    L3group = df_grouped_by_id_train['L3']
    L4group = df_grouped_by_id_train['L4']
    L5group = df_grouped_by_id_train['L5']
    L6group = df_grouped_by_id_train['L6']
    L7group = df_grouped_by_id_train['L7']
    L8group = df_grouped_by_id_train['L8']
    L9group = df_grouped_by_id_train['L9']
    L10group = df_grouped_by_id_train['L10']
    L11group = df_grouped_by_id_train['L11']
    L12group = df_grouped_by_id_train['L12']
    L14group = df_grouped_by_id_train['L14']
    L15group = df_grouped_by_id_train['L15']
    #L16group = df_grouped_by_id_train['L16']
    L18group = df_grouped_by_id_train['L18']
    L19group = df_grouped_by_id_train['L19']
    L20group = df_grouped_by_id_train['L20']
    L21group = df_grouped_by_id_train['L21']
    #L22group = df_grouped_by_id_train['L22']
    L23group = df_grouped_by_id_train['L23']
    #L24group = df_grouped_by_id_train['L24']
    L25group = df_grouped_by_id_train['L25']
    V1group = df_grouped_by_id_train['V1']
    V2group = df_grouped_by_id_train['V2']
    V3group = df_grouped_by_id_train['V3']
    V4group = df_grouped_by_id_train['V4']
    V5group = df_grouped_by_id_train['V5']
    V6group = df_grouped_by_id_train['V6']
 
 
 
    df_L1 = L1group.agg( {Lnames[1]+suffixes[3] : np.min} ).reset_index()
    df_L2 = L2group.agg( {Lnames[2]+suffixes[5] : np.std } ).reset_index()
    x = pd.merge( df_L1, df_L2, on = 'ID')
 
 
    df_L3 = L3group.agg( { Lnames[3]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L3, on = 'ID')
 
    df_L4 = L4group.agg( { Lnames[4]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L4, on = 'ID')
 
    df_L5 = L5group.agg( { Lnames[5]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L5, on = 'ID')
 
    df_L6 = L6group.agg( { Lnames[6]+suffixes[0] : lambda x: f_last_for_series(x), Lnames[6]+suffixes[3] : np.min } ).reset_index()
    x = pd.merge( x, df_L6, on = 'ID')
 
    df_L7 = L7group.agg( { Lnames[7]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L7, on = 'ID')
 
    df_L8 = L8group.agg( { Lnames[8]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L8, on = 'ID')
 
    df_L9 = L9group.agg( { Lnames[9]+suffixes[3] : np.min } ).reset_index()
    x = pd.merge( x, df_L9, on = 'ID')
 
    df_L10 = L10group.agg( { Lnames[10]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L10, on = 'ID')
 
    df_L11 = L11group.agg( { Lnames[11]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L11, on = 'ID')
 
    df_L12 = L12group.agg( { Lnames[12]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L12, on = 'ID')
 
    df_L14 = L14group.agg( { Lnames[14]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) } ).reset_index()
    x = pd.merge( x, df_L14, on = 'ID')
 
    df_L15 = L15group.agg( { Lnames[15]+suffixes[0] : lambda x: f_last_for_series(x), Lnames[15]+suffixes[2] : np.mean } ).reset_index()
    x = pd.merge( x, df_L15, on = 'ID')
 
    df_L18 = L18group.agg( { Lnames[18]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) } ).reset_index()
    x = pd.merge( x, df_L18, on = 'ID')
 
    df_L19 = L19group.agg( { Lnames[19]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L19, on = 'ID')
 
    df_L20 = L20group.agg( { Lnames[20]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L20, on = 'ID')
 
    df_L21 = L21group.agg( { Lnames[21]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L21, on = 'ID')
 
    df_L23 = L23group.agg( { Lnames[23]+suffixes[7] : np.max } ).reset_index()
    x = pd.merge( x, df_L23, on = 'ID')
 
    df_L25 = L25group.agg( { Lnames[25]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L25, on = 'ID')
 
    df_V1 = V1group.agg( { Vnames[1]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[1]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[1]+suffixes[2] : np.mean, Vnames[1]+suffixes[3] : np.min} ).reset_index()
    x = pd.merge( x, df_V1, on = 'ID')
 
    df_V2 = V2group.agg( { Vnames[2]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[2]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[2]+suffixes[3] : np.min, Vnames[2]+suffixes[5] : np.std, Vnames[2]+suffixes[6] : lambda x: f_ratio_for_series(x)} ).reset_index()
    x = pd.merge( x, df_V2, on = 'ID')
 
    df_V3 = V3group.agg( { Vnames[3]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[3]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[3]+suffixes[3] : np.min} ).reset_index()
    x = pd.merge( x, df_V3, on = 'ID')
 
    df_V4 = V4group.agg( { Vnames[4]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[4]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[4]+suffixes[2] : np.mean, Vnames[4]+suffixes[3] : np.min, Vnames[4]+suffixes[5] : np.std} ).reset_index()
    x = pd.merge( x, df_V4, on = 'ID')
 
    df_V5 = V5group.agg( { Vnames[5]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[5]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[5]+suffixes[2] : np.mean, Vnames[5]+suffixes[5] : np.std} ).reset_index()
    x = pd.merge( x, df_V5, on = 'ID')
 
    df_V6 = V6group.agg( { Vnames[6]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[6]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[6]+suffixes[3] : np.min} ).reset_index()
    x = pd.merge( x, df_V6, on = 'ID')
 
 
 
    x = pd.merge( x, df_age_train, on = 'ID')  # appending age
 
 
    #finally the labels
    df_labels_train = pd.read_csv('id_label_train.csv')
    x = pd.merge(x, df_labels_train , on = 'ID') 
 
    #print x.columns.values
    #so y is our final dataframe which will be passed to the featue selection algorithms
    y  =  fill_normal_in_missing(x)
    #print y[y.isnull() == True]
    return y
 
 
def feature_extraction_val():
    """
    will return a pd.Dataframe with 291 features extracted
    from time series as columns and 2 columns more i.e. ID and age.
    So total 293 columns
    """
    df_without_time_val = df_variables_combined_val.drop(['TIME'], axis=1)
    df_grouped_by_id_val = df_without_time_val.groupby('ID')
 
    L1group = df_grouped_by_id_val['L1']
    L2group = df_grouped_by_id_val['L2']
    L3group = df_grouped_by_id_val['L3']
    L4group = df_grouped_by_id_val['L4']
    L5group = df_grouped_by_id_val['L5']
    L6group = df_grouped_by_id_val['L6']
    L7group = df_grouped_by_id_val['L7']
    L8group = df_grouped_by_id_val['L8']
    L9group = df_grouped_by_id_val['L9']
    L10group = df_grouped_by_id_val['L10']
    L11group = df_grouped_by_id_val['L11']
    L12group = df_grouped_by_id_val['L12']
    L14group = df_grouped_by_id_val['L14']
    L15group = df_grouped_by_id_val['L15']
    #L16group = df_grouped_by_id_val['L16']
    L18group = df_grouped_by_id_val['L18']
    L19group = df_grouped_by_id_val['L19']
    L20group = df_grouped_by_id_val['L20']
    L21group = df_grouped_by_id_val['L21']
    #L22group = df_grouped_by_id_val['L22']
    L23group = df_grouped_by_id_val['L23']
    #L24group = df_grouped_by_id_val['L24']
    L25group = df_grouped_by_id_val['L25']
    V1group = df_grouped_by_id_val['V1']
    V2group = df_grouped_by_id_val['V2']
    V3group = df_grouped_by_id_val['V3']
    V4group = df_grouped_by_id_val['V4']
    V5group = df_grouped_by_id_val['V5']
    V6group = df_grouped_by_id_val['V6']
 
 
 
 
 
    df_L1 = L1group.agg( {Lnames[1]+suffixes[3] : np.min} ).reset_index()
    df_L2 = L2group.agg( {Lnames[2]+suffixes[5] : np.std } ).reset_index()
    x = pd.merge( df_L1, df_L2, on = 'ID')
 
 
    df_L3 = L3group.agg( { Lnames[3]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L3, on = 'ID')
 
    df_L4 = L4group.agg( { Lnames[4]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L4, on = 'ID')
 
    df_L5 = L5group.agg( { Lnames[5]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L5, on = 'ID')
 
    df_L6 = L6group.agg( { Lnames[6]+suffixes[0] : lambda x: f_last_for_series(x), Lnames[6]+suffixes[3] : np.min } ).reset_index()
    x = pd.merge( x, df_L6, on = 'ID')
 
    df_L7 = L7group.agg( { Lnames[7]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L7, on = 'ID')
 
    df_L8 = L8group.agg( { Lnames[8]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L8, on = 'ID')
 
    df_L9 = L9group.agg( { Lnames[9]+suffixes[3] : np.min } ).reset_index()
    x = pd.merge( x, df_L9, on = 'ID')
 
    df_L10 = L10group.agg( { Lnames[10]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L10, on = 'ID')
 
    df_L11 = L11group.agg( { Lnames[11]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L11, on = 'ID')
 
    df_L12 = L12group.agg( { Lnames[12]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L12, on = 'ID')
 
    df_L14 = L14group.agg( { Lnames[14]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) } ).reset_index()
    x = pd.merge( x, df_L14, on = 'ID')
 
    df_L15 = L15group.agg( { Lnames[15]+suffixes[0] : lambda x: f_last_for_series(x), Lnames[15]+suffixes[2] : np.mean } ).reset_index()
    x = pd.merge( x, df_L15, on = 'ID')
 
    df_L18 = L18group.agg( { Lnames[18]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) } ).reset_index()
    x = pd.merge( x, df_L18, on = 'ID')
 
    df_L19 = L19group.agg( { Lnames[19]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L19, on = 'ID')
 
    df_L20 = L20group.agg( { Lnames[20]+suffixes[6] : lambda x: f_ratio_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L20, on = 'ID')
 
    df_L21 = L21group.agg( { Lnames[21]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L21, on = 'ID')
 
    df_L23 = L23group.agg( { Lnames[23]+suffixes[7] : np.max } ).reset_index()
    x = pd.merge( x, df_L23, on = 'ID')
 
    df_L25 = L25group.agg( { Lnames[25]+suffixes[0] : lambda x: f_last_for_series(x) } ).reset_index()
    x = pd.merge( x, df_L25, on = 'ID')
 
    df_V1 = V1group.agg( { Vnames[1]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[1]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[1]+suffixes[2] : np.mean, Vnames[1]+suffixes[3] : np.min} ).reset_index()
    x = pd.merge( x, df_V1, on = 'ID')
 
    df_V2 = V2group.agg( { Vnames[2]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[2]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[2]+suffixes[3] : np.min, Vnames[2]+suffixes[5] : np.std, Vnames[2]+suffixes[6] : lambda x: f_ratio_for_series(x)} ).reset_index()
    x = pd.merge( x, df_V2, on = 'ID')
 
    df_V3 = V3group.agg( { Vnames[3]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[3]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[3]+suffixes[3] : np.min} ).reset_index()
    x = pd.merge( x, df_V3, on = 'ID')
 
    df_V4 = V4group.agg( { Vnames[4]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[4]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[4]+suffixes[2] : np.mean, Vnames[4]+suffixes[3] : np.min, Vnames[4]+suffixes[5] : np.std} ).reset_index()
    x = pd.merge( x, df_V4, on = 'ID')
 
    df_V5 = V5group.agg( { Vnames[5]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[5]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[5]+suffixes[2] : np.mean, Vnames[5]+suffixes[5] : np.std} ).reset_index()
    x = pd.merge( x, df_V5, on = 'ID')
 
    df_V6 = V6group.agg( { Vnames[6]+suffixes[0] : lambda x: f_last_for_series(x),  Vnames[6]+suffixes[1] : lambda x: f_exp_avg_for_series(x, alpha) ,Vnames[6]+suffixes[3] : np.min} ).reset_index()
    x = pd.merge( x, df_V6, on = 'ID')
 
    x = pd.merge( x, df_age_val, on = 'ID')  # appending age
 
 
    #finally the labels
    df_labels_val = pd.read_csv('id_label_val.csv')
    x = pd.merge(x, df_labels_val , on = 'ID') 
 
    #print x.columns.values
    #so y is our final dataframe which will be passed to the featue selection algorithms
    y  =  fill_normal_in_missing(x)
    #print y[y.isnull() == True]
    return y
 

df_train = feature_extraction_train() 
X_train = df_train.drop(['ID', 'LABEL'], axis=1)
Y_train = df_train[ 'LABEL' ].values
 
df_val = feature_extraction_val()  
X_val = df_val.drop(['ID', 'LABEL'], axis=1)
Y_val = df_val[ 'LABEL' ].values
 
actual_Y_val = Y_val.tolist()


"""
rf = RandomForestClassifier(random_state=0)
 
csrf = CostSensitiveRandomForestClassifier(n_estimators = 10,
                                           combination = 'majority_voting',
                                           max_features = None,
                                           pruned = True
                                           )

n_estimators range from 10 to 100. Default 10
 
combination_list = [ 'majority_voting', 'weighted_voting', 'stacking',
'stacking_proba', 'stacking_bmr', 'stacking_proba_bmr', 'majority_bmr',
'weighted_bmr' ]
 

n = 3594
cost_mat = np.ones((n,4))
cost_mat[:,0] = cost_mat[:,0] * 5
cost_mat[:,1] = cost_mat[:,1] * 1
cost_mat[:,2] = cost_mat[:,2] * 0
cost_mat[:,3] = cost_mat[:,3] * 0 
 
cost_mat_train = cost_mat
cost_mat_test = cost_mat

rf.fit(X_train, Y_train)
y_pred_test_rf = rf.predict(X_val)
y_prob_test_rf = rf.predict_proba(X_val)
y_prob_train_rf = rf.predict_proba(X_train)
 
csrf.fit(X_train, Y_train, cost_mat_train)
y_pred_test_csrf = csrf.predict(X_val, cost_mat_test)
y_prob_test_csrf = csrf.predict_proba(X_val)
y_prob_train_csrf = csrf.predict_proba(X_train)

"""

"""
By default, weak learners are decision stumps. Different weak learners can be specified through the base_estimator parameter. The main parameters to tune to obtain good results are n_estimators and the complexity of the base estimators (e.g., its depth max_depth or minimum required number of samples at a leaf min_samples_leaf in case of decision trees).

from sklearn.ensemble import AdaBoostClassifier
output_list = []
for i in range(1,8):
    #print 'min_samples_leaf = ' , i
    for no_of_estimators in [50 , 75 , 100 , 125 , 150]:
        #print '\nNo. of estimators = ',no_of_estimators
        clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(min_samples_leaf =5) , n_estimators=no_of_estimators)   
        clf.fit(X_train , Y_train )
        predicted_y_clf = clf.predict(X_val).tolist()
 
        specificity_clf, sensitivity_clf, fp_clf, fn_clf, tp_clf, tn_clf = get_results(predicted_y_clf , actual_Y_val)
# specificity_rf, sensitivity_rf, fp_rf, fn_rf, tp_rf, tn_rf = get_results(predicted_y_rf , actual_Y_val)
# specificity_csrf, sensitivity_csrf, fp_csrf, fn_csrf, tp_csrf, tn_csrf = get_results(predicted_y_csrf , actual_Y_val)
        output_list.append([i , no_of_estimators , specificity_clf, sensitivity_clf, fp_clf, fn_clf, tp_clf, tn_clf])
        #print 'Specificity = ' , specificity_clf, '\nSensitivity = ' , sensitivity_clf, '\nFP = ' , fp_clf, '\nFN = ', fn_clf, '\nTP = ' , tp_clf, '\nTN = ', tn_clf
"""
output_list = []
for maxdepth in range(3 , 10):
    for i in [100 ,125 , 150 , 175 , 200 , 225 ]:
        gbc = GradientBoostingClassifier(n_estimators = i , max_depth = maxdepth)
        gbc.fit(X_train , Y_train)
        predicted_y_gbc = gbc.predict(X_val).tolist()
        specificity_gbc, sensitivity_gbc, fp_gbc, fn_gbc, tp_gbc, tn_gbc = get_results(predicted_y_gbc , actual_Y_val)
        output_list.append([maxdepth , i , specificity_gbc, sensitivity_gbc, fp_gbc, fn_gbc, tp_gbc, tn_gbc])
        #print 'Specificity = ' , specificity_gbc, '\nSensitivity = ' , sensitivity_gbc, '\nFP = ' , fp_gbc, '\nFN = ', fn_gbc, '\nTP = ' , tp_gbc, '\nTN = ', tn_gbc

wr = csv.writer(open('GradientBoostingClassifier_results.csv' , 'wb'))
wr.writerow(['max_depth' , 'n_estimators' , 'Sp' , 'Se' , 'FP' , 'FN' , 'TP' , 'TN'])
wr.writerows(output_list)