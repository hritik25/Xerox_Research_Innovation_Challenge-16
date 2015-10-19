import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

header = csv.reader(open('id_variables_combined_train.csv' , 'rb'))
df= pd.read_csv('id_variables_combined_train.csv')
labels = header.next()

#print labels

x = []
y = []

for i in range(2,9):

	timestamp_bins = {}
	f = csv.reader(open('id_variables_combined_train.csv' , 'rb'))
	next(f)
	#print f

	#print df[labels[i]].min() , df[labels[i]].max()

	for row in f :
		if row[i]!='NA':
			timestamp_bins.setdefault(int(row[1])/36000 , []).append(float(row[i]))

	x.append([])
	y.append([])

	for key in timestamp_bins:
		timestamp_bins[key] = np.mean(timestamp_bins[key])
		x[i-2].append(key)
		y[i-2].append((timestamp_bins[key] - df[labels[i]].min())/(df[labels[i]].max() - df[labels[i]].min()))

	#print x[i-2]
	#break
	
	plt.xticks(np.arange(min(x[i-2]) , max(x[i-2])+10 , 50))
	plt.xlabel('x 10 hours')
	plt.ylabel('Value b/w 0 and 1')
	plt.plot(x[i-2] , y[i-2] , label = labels[i])

mpl.rcParams['axes.color_cycle'] =
		['b',
        'g',
        'r',
        'c',
        'm',
        'y',
        'k',
        'w',

]
plt.legend()
plt.show()
	 
#print timestamp_bins """