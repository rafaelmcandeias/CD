import numpy as np
from numpy.core.numerictypes import typecodes
from ds_charts import get_variable_types, bar_chart
from pandas import concat, DataFrame, read_csv
import pandas as pd
from sklearn.impute import SimpleImputer
from matplotlib.pyplot import figure, savefig
from pandas.plotting import register_matplotlib_converters



# ------------- FIRST DATASET --------------------------------
register_matplotlib_converters()
filepath = '../../data/firstDataset/NYC_collisions_tabular_no_mvs.csv'
data = read_csv(filepath, na_values='', parse_dates=True, infer_datetime_format=True)
df = pd.DataFrame(data)


# ------------- INFO ------------------------------------------
#print(df.info)


# ------------- UNIQUES ---------------------------------------
#for f in df:
#    print(f, df[f].unique(), '\n')


# ------------- PERSON_TYPES COM MV EM VEHICLE_ID -------------
#vi = df['VEHICLE_ID'].values.tolist()
#pt = df['PERSON_TYPE'].values.tolist()
#ci = df['COLLISION_ID'].values.tolist()
#types = set()
#count_p, count_o = 0, 0
#for i in range(len(vi)):
#    if pd.isnull(vi[i]) and pt[i] == 'Pedestrian':
#        count_p += 1
#    if pd.isnull(vi[i]) and pt[i] == 'Occupant':
#        count_o += 1
#        for j in range(len(ci)):
#            if ci[j] == ci[i]:
#                if not pd.isnull(vi[j]):
#                    print(vi[j])
#print(count_p, count_o)


# -- PEDESTRIAN AND OCCUPANT HAVE MV AND VALUES IN EJECTION ---
e = df['EJECTION'].values.tolist()
pt = df['PERSON_TYPE'].values.tolist()
types = set()
for i in range(len(e)):
    if pd.isnull(e[i]):
        types.add(pt[i])
print(types)


# ------------- PERSON_TYPES COM MV EM POSITION_IN_VEHICLE ----
pv = df['POSITION_IN_VEHICLE'].values.tolist()
pt = df['PERSON_TYPE'].values.tolist()
types = set()
count_p, count_o = 0, 0
for i in range(len(pv)):
    if pd.isnull(pv[i]) and pt[i] == 'Pedestrian':
        count_p += 1
    if pd.isnull(pv[i]) and pt[i] == 'Occupant':
        count_o += 1
print(count_p, count_o)


# ------------- DESCOBRE OS PARES DE CF1 e CF2 ---------------- 
# verificar se CONTRIBUTING_FACTOR_1 tem valores nao nulos != dos valores nao nulos do  
#cf1 = df['CONTRIBUTING_FACTOR_1'].values.tolist()
#cf2 = df['CONTRIBUTING_FACTOR_2'].values.tolist()
#pt = df['PERSON_TYPE'].values.tolist()
#diff = set()
# cria set de tuplos com os cf1 e cf2 nao nulos nem unspecified e diferentes entre si
#for i in range(len(cf1)):
#    if not pd.isnull(cf1[i]) and not pd.isnull(cf2[i]) and cf1[i] != cf2[i]:
#        if cf1[i] != 'Unspecified' and cf2[i] != 'Unspecified':
#            diff.add((pt[i], cf1[i], cf2[i]))
#for v in diff:
#    print(v)


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


# ------------- SAFETY EQUIPMENT EM PEDESTRIANS? ----------------------------
#se = df['SAFETY_EQUIPMENT'].values.tolist()
#pt = df['PERSON_TYPE'].values.tolist()
#unique = set()
#for i in range(len(se)):
#    if pt[i] == 'Pedestrian':
#        unique.add(se[i])
#print(unique)



# ------------- SECOND DATASET --------------------------------
#register_matplotlib_converters()
#filepath = '../../data/secondDataset/air_quality_tabular.csv'
#data = read_csv(filepath, index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)
#df = data
