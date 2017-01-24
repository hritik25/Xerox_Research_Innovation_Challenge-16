"""
scoring function for self evaluation 
"""

# output file in the following format (DEFINED ONLY FOR ICU LABEL 1)
# Patient ID,Prediction timestamp,Prediction at that timestamp\n
import pandas as pd

#lets make three dataframes :
#1.first with the maxes of icu labels grouped by ID , this would serve as predictions for the patients
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
	# group this temp dataframe by ID and drop the TIME stamp column , as this dataframe contains the first timestamp where a patient's death was predicted , if it was
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
	temp = first_prediction_of_death_df.merge(last_timestamp_df , on = ['ID'] , how  = 'inner' , suffixes = ('_first_timestamp_of_one' , '_last_timestamp'))
	temp['PREDICTION_TIME'] = temp['TIME__last_timestamp'] - temp['TIME_first_timestamp_of_one']
	median_prediction_time = temp['PREDICTION_TIME'].median()
	return predicted_labels , median_prediction_time
		


"""
given two lists , one with predicted labels , one with actual labels
calculate calculate the no. of TNs , TPs , FPs , FNs and hence the specificity and sensitivity of predictions
"""

def get_results():
	#the four parameters of evaluation
	actual_labels_df = pd.read_csv(open('id_label_val.csv')) #filename hardcoded
	actual = np.array(actual_labels_df['LABEL']).tolist()
	predicted , median_prediction_time = get_predicted_labels_and_median_time():
	tn = 0
	tp = 0
	fn = 0
	fp = 0
	for i in range(len(predicted)):
		if predicted[i] == 1 & actual[i] == 1:
			tp = tp + 1
		elif predicted[i] == 1 & actual[i] == 0:
			fp = fp + 1
		elif predicted[i] == 0 & actula[i] == 0:
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

	return specificity , sensitivity , median_prediction_time

def evaluate_score():
	#get the following values 
	specificity , sensitivity , fp , fn , tp , tn = get_results()
	
	median_prediction_time = (float)median_prediction_time/3600
	
	if median_prediction_time < 72.0:
		median_prediction_time_clipped_at_72 = median_prediction_time
	else:
		median_prediction_time_clipped_at_72 = 72 
	
	if specificity < 0.0:
		return -1
	elif sensitivity == 0.0:
		return -2
	elif median_prediction_time_clipped_at_72 < 5.0 :
		return -3
	else:
		median_prediction_time_score = (float)median_prediction_time_clipped_at_72 / 72
		final_score = 0.75*sensitivity + 0.2*median_prediction_time_score + 0.05*specificity
		return final_score

