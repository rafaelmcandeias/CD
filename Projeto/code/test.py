import numpy as np
from ds_charts import get_variable_types, bar_chart
from pandas import concat, DataFrame, read_csv
import pandas as pd
from sklearn.impute import SimpleImputer
from matplotlib.pyplot import figure, savefig
from pandas.plotting import register_matplotlib_converters



# ------------- FIRST DATASET --------------------------------
register_matplotlib_converters()
filepath = '../../data/firstDataset/NYC_collisions_tabular.csv'
data = read_csv(filepath, index_col='UNIQUE_ID', na_values='', parse_dates=True, infer_datetime_format=True)
df = data


# ------------- DESCOBRE OS PARES DE CF1 e CF2 ---------------- 
# verificar se CONTRIBUTING_FACTOR_1 tem valores nao nulos != dos valores nao nulos do  
#cf1 = df['CONTRIBUTING_FACTOR_1'].values.tolist()
#cf2 = df['CONTRIBUTING_FACTOR_2'].values.tolist()
#diff = set()
# cria set de tuplos com os cf1 e cf2 nao nulos nem unspecified e diferentes entre si
#for i in range(len(cf1)):
#    if not pd.isnull(cf1[i]) and not pd.isnull(cf2[i]) and cf1[i] != cf2[i]:
#        if cf1[i] != 'Unspecified' and cf2[i] != 'Unspecified':
#            diff.add((cf1[i], cf2[i]))
#for v in diff:
    # print(v)
#print(len(diff))


# ------------- MV DE CF1 E CF2 PARA PEDESTRIANS --------------
#cf1 = df['CONTRIBUTING_FACTOR_1'].values.tolist()
#cf2 = df['CONTRIBUTING_FACTOR_2'].values.tolist()
#pt = df['PERSON_TYPE'].values.tolist()
#mv1, mv2 = 0, 0
#for idx in range(len(cf1)):
#    if pd.isnull(cf1[idx]) and pt[idx] == 'Pedestrian': 
        #mv1 += 1
#    if pd.isnull(cf2[idx]) and pt[idx] == 'Pedestrian':
        #mv2 += 1
#print(mv1, mv2)


# ------------- CRASH_TIME IMPORTANTE? ------------------------
#time = df['CRASH_TIME'].values.tolist()
#target = df['PERSON_INJURY'].values.tolist()
#hours = ('00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
#         '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00')
#times = dict.fromkeys(hours, 0)
#n_records = df.shape[0
#for idx in range(n_records):
#    if target[idx] == 'Killed':
#        times[time[idx]] += 1
#sorted = dict(sorted(times.items(), key=lambda item: item[1]))
#for key in sorted.keys():
#    print(key, ":", sorted[key]/n_records)


# ------------- DUMMIFICATION? --------------------------------
#print(df['EJECTION'].unique())
#print(len(df['SAFETY_EQUIPMENT'].unique()))



# ------------- SECOND DATASET --------------------------------
#register_matplotlib_converters()
#filepath = '../../data/secondDataset/air_quality_tabular.csv'
#data = read_csv(filepath, index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)
#df = data
