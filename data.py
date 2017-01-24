#describe data
import numpy as np
import pandas as pd

df = pd.read_csv('id_variables_combined_train.csv')

print df.describe()