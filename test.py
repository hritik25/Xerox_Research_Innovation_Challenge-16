import pandas as pd
import numpy as np
import csv

"""
labs = []
vitals = []

csv_f = csv.reader(open('id_time_labs_train.csv'))

for row in csv_f:
	labs.append(row)

csv_f  = csv.reader(open('id_time_vitals_train.csv'))

for row in csv_f:
	vitals.append(row[2:])

#print len(labs) , len(vitals)

labs_vitals_combined = []

for i in range(0,len(labs)):
	labs_vitals_combined.append(labs[i]+vitals[i])

print len(labs_vitals_combined)

with open('id_variables_combined_train.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(labs_vitals_combined)

f.close()

"""
ages = []
labels = []

csv_f = csv.reader(open('id_age_train.csv'))

for row in csv_f:
	ages.append(row)

csv_f  = csv.reader(open('id_label_train.csv'))

for row in csv_f:
	labels.append([row[1]])

#print len(labs) , len(vitals)

age_label_train = []

for i in range(0,len(ages)):
	age_label_train.append(ages[i]+labels[i])


with open('id_age_label_train.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(age_label_train)

f.close() 

