import pandas as pd
import csv
import numpy as np

df = pd.read_csv('output_gbc.csv')
df = df[df['PREDICTION'] == 1].groupby('ID').first().reset_index()
df = df[['ID']]
x = np.array(df['ID'])
w = csv.writer(open('list.csv','wb'))
w.writerow([x])
