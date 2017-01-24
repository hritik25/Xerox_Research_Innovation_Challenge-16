"""
code for online prediciton 
"""
import pandas as pd
import csv
import sys

#i am assuming we have a dataframe with vitals and lab investigations combined , lets call it 'variables_combined_df'
#so first i get min and max patient ID in 'variables_combined_df'
#also assuming the min() and max() of patient IDs is with us , not hardcoded , but through the file 'id_age_test.csv'
#they are in the variables min_id and max_id

# an empty list of outputs , which will be converted to the file named 'output.csv'
id_time_vitals = sys.argv[1]
id_time_labs = sys.argv[2]
id_age = sys.argv[3]

df_labs_test = pd.read_csv(id_time_labs)
df_vitals_test = pd.read_csv(id_time_vitals)
df_age_test = pd.read_csv(id_age)

df_variables_combined_test = df_labs_test.merge(df_vitals_test , on = ['ID','TIME'])    
df_variables_combined_test = df_variables_combined_test.drop(['L13', 'L17'], axis=1)  # Dropping the columns of L13 and L17
df_variables_combined_test['L20'] = df_variables_combined_test['L3']/df_variables_combined_test['L20']

output = []
for i in range(min_id , max_id):
	grouped_by_id = df_variables_combined_test.groupby('ID')
 	current_patient_df = grouped_by_id.get_group(i)
	for j,row in current_patient_df.iterrows():
		#lets assume the index of ICU in the dataframe we have is 'index'
		#start predicting if the patient goes to ICU
		"""Here DON'T forget to append L15 , L22 to the last of the np.array you will be passing to the model , just do np.append """
		flag = 0:
		if row[] == 1:
			if not flag:
				l15_l22_first_in_icu = (row[15] , row[19])
				flag = 1
			data_so_far_df = current_patient_df.head(j+1)
			#pass this data to the function which will do stuff with this and return a np.array which is supposed to be the 
			#input feature vector for our classifier model
			feature_value_vector = get_patient_feature_values(i,data_so_far_df.drop(['ID','TIME'],axis = 1))
			feature_value_vector = np.append(feature_value_vector , l15_l22_first_in_icu)

			"""
			Space left for the classifierode of prediciton .
			At the end of this code , we will have a value as 0 or 1 which is the PREDICTION for this patient for this Timestamp
			Let's have it returned to the variable 'predicted'
			"""
			output.append([row[0] , row[1] , predicted])

#now write the output to a csv file
wr  = csv.writer(open('output.csv' , 'wb'))
""" this is for the header , just for your own evaluation , but DON'T include this for Submission """
wr.writerows(['ID' , 'TIME' , 'PREDICTION'])
wr.writerows(output)	


