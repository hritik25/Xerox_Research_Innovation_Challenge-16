import pandas as pd

df = pd.read_csv('output.csv')
groupby_id = df[df['PREDICTION']==1].groupby('ID')
dfn_first = groupby_id.first().reset_index()
print dfn_first.shape
dfn_last =df.groupby('ID').last().reset_index()
print dfn_last.shape
dfn_combined = dfn_first.merge(dfn_last , on = ['ID'] , how = 'inner' , suffixes = ('_first' , '_last') )
print dfn_combined.shape
dfn_combined['prediction_time'] = dfn_combined['TIME_last'] - dfn_combined['TIME_first']
print dfn_combined[['ID','prediction_time']].sort('prediction_time')
print 'median prediction time' , dfn_combined['prediction_time'].median()

