import pandas as pd
import csv

df = pd.read_csv('output.csv')
x = df
x = x.astype(int)

w = csv.writer('output_int.csv' , 'wb')
for row in x.itertuples():
	w.writerow(row[1:])

