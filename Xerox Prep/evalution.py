"""
given two lists , one with predicted labels , one with actual labels
calculate calculate the no. of TNs , TPs , FPs , FNs and hence the specificity and sensitivity of predictions
"""

def get_results(predicted , actual):
	#the four parameters of evaluation
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
	accuracy = float(tp + tn) / (tp + tn + fp + fn)

	return specificity , sensitivity , accuracy 
