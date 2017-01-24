import csv
import numpy as np
import matplotlib.pyplot as plt

f = csv.reader(open('id_variables_combined_train.csv' , 'rb'))

x = []
y=[]

for row in f:
	if row[0] == '5':
		if row[5]!='NA':
			y.append(float(row[5]))
			x.append(int(row[1])/3600)

plt.xticks(np.arange(min(x) , max(x)+1 , 100))
plt.ylabel('Amount of sodium in mmol/L')
plt.xlabel('Time stamps in hours')
plt.title('Sodium variation for Patient 5')
plt.plot(x , y)
plt.show()