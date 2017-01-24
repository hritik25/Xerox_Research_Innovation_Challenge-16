""" 
final code's input section
taking as command line parameters the test data files in the following order:
1.	id_time_vitals ( like 'id_time_vitals_test.csv' )
2.	id_time_labs ( like 'id_time_labs_test.csv')
3.	id_age ( like ' id_age_test.csv ') 
"""

import sys
import pandas as pd

id_time_vitals = sys.argv[1]
id_time_labs = sys.argv[2]
id_age = sys.argv[3]

df_labs_test = pd.read_csv(id_time_labs)
df_vitals_test = pd.read_csv(id_time_vitals)
df_age_test = pd.read_csv(id_age)

df_variables_combined_test = df_labs_test.merge(df_vitals_test , on = ['ID','TIME'])    
df_variables_combined_test = df_variables_combined_test.drop(['L13', 'L17'], axis=1)  # Dropping the columns of L13 and L17
df_variables_combined_test['L20'] = df_variables_combined_test['L3']/df_variables_combined_test['L20']