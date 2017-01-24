import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#print labels



fig = plt.figure()

f = csv.reader(open('id_variables_combined_train.csv' , 'rb'))
labels = f.next()
	#print f
	#print df[labels[i]].min() , df[labels[i]].max()
for i in range(2,10):

	timestamp_bins = {'1': {} , '0':{} }
	x0= []
	y0= []

	x1 = []
	y1 = []

	for row in f :
		if row[i]!='NA':
			if row[33] == '1':
				timestamp_bins['1'].setdefault(int(row[1])/36000 , []).append(float(row[i]))
			else :
				timestamp_bins['0'].setdefault(int(row[1])/36000 , []).append(float(row[i]))

	for keyy in timestamp_bins['1']:
		timestamp_bins['1'][keyy] = np.mean(timestamp_bins['1'][keyy])
		x1.append(keyy)
		y1.append(timestamp_bins['1'][keyy])

	for key in timestamp_bins['0']:

		timestamp_bins['0'][key] = np.mean(timestamp_bins['0'][key])
		x0.append(key)
		y0.append(timestamp_bins['0'][key])

		ax1 = fig.add_subplot(8,1,i-1)
		ax1.plot(x1 , y1 , label = 'Died')
		ax1.plot(x0 , y0 , label = 'Survived')

plt.title(labels[i])
#plt.xticks(np.arange(min(x0), max(x0)+10 , 50))
plt.xlabel('(x 10 hours) = Timestamp')
plt.ylabel('Value b/w 0 and 1')

plt.legend()
plt.show()