import pandas as pd

dfn = pd.read_csv('output.csv')
dfn = dfn[dfn['PREDICTION'] == 1]
groupby_id = dfn[dfn['PREDICTION']==1].groupby('ID')
dfn_first = groupby_id.first().reset_index()
dfn_last = groupby_id.last().reset_index()
dfn_combined = dfn_first.merge(dfn_last , on = ['ID'] , suffixes = ('_first' , '_last'))
dfn_combined['prediction_time'] = dfn_combined['TIME_last'] - dfn_combined['TIME_first']
print 'median prediction time' ,dfn_combined['prediction_time'].median()

