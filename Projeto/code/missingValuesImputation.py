from numpy import nan
from ds_charts import get_variable_types
from pandas import concat, DataFrame, read_csv
from sklearn.impute import SimpleImputer
from pandas.plotting import register_matplotlib_converters


# PERSON_SEX and PERSON_TYPE have no mvs
nan_constant_fill = -1
# PED_LOCATION and POSITION_IN_VEHICLE mvs fill
# for PED_LOCATION we only have to fill mvs with a constant, because every mv is from when PERSON_TYPE!= Pedestrain
df.loc[df['PED_LOCATION'].isna(), 'PED_LOCATION'] = nan_constant_fill

#PED_ACTION == nan when PERSON_TYPE == Pedestrian
df.loc[df['PED_ACTION'].isna(), 'PED_ACTION'] = nan_constant_fill


# ------------- FIRST DATASET ---------------------------------
register_matplotlib_converters()
filepath = '../../data/firstDataset/NYC_collisions_tabular.csv'
data = read_csv(filepath, index_col='UNIQUE_ID', na_values='', parse_dates=True, infer_datetime_format=True)
df = DataFrame(data)

# Numero de missing values por variavel
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr
print(mv)

# Gets most frequent BODILY_INJURY for each SAFETY_EQUIPMENT 
# (it's always BODILY_INJURY = lap Belt & Harness except 
# for SAFETY_EQUIPMENT = Knee-Lower Leg Foot, where BODILY_INJURY = None)
sfty = df['SAFETY_EQUIPMENT']
bdly = df['BODILY_INJURY']
values = df.groupby('BODILY_INJURY')['SAFETY_EQUIPMENT'].apply(
    lambda x: x.value_counts().index[0]).reset_index()

# fills mvs
df.loc[df['SAFETY_EQUIPMENT'].isna() & df['BODILY_INJURY'] ==
        "Knee-Lower Leg Foot", 'SAFETY_EQUIPMENT'] = "None"
df.loc[df['SAFETY_EQUIPMENT'].isna(), 'SAFETY_EQUIPMENT'] = "Lap Belt & Harness"

# For EJECTION, Pedestrians shouldn't have and most are missing. 
df.loc[df['EJECTION'].isna() & df['PERSON_TYPE'] ==
        "Pedestrian", 'EJECTION'] = nan_constant_fill
# Occupants tem poucos mv, usa se most_frequent
df.loc[df['EJECTION'].isna() & df['PERSON_TYPE'] == 
        "Occupant", 'EJECTION'] = df['EJECTION'].mode()

# Puts all VEHICLE_ID in Pedestrian as nan since it doenst make sence to associate both
df.loc[df['VEHICLE_ID'].isna() & df['PERSON_TYPE'] ==
        "Pedestrian" , 'VEHICLE_ID'] = nan_constant_fill
# Puts all VEHICLE_ID in Occupant as the value from another register with the same COLLISION_ID? 
# Only 9% isn't nan, even from another register. Not worth it. 
# Apply most_frequent instead
df.loc[df['VEHICLE_ID'].isna() & df['PERSON_TYPE'] ==
       "Occupant", 'VEHICLE_ID'] = df['VEHICLE_ID'].mode()

#For POSITION_IN_VEHICLE the same thing
df.loc[df['POSITION_IN_VEHICLE'].isna() & df['PERSON_TYPE'] ==
        "Pedestrian", 'POSITION_IN_VEHICLE'] = nan_constant_fill
df.loc[df['POSITION_IN_VEHICLE'].isna() & df['PERSON_TYPE'] ==
       "Occupant", 'POSITION_IN_VEHICLE'] = df['POSITION_IN_VEHICLE'].mode()



# Filling aproaches:
tmp_nr, tmp_sb, tmp_bool = None, None, None
# Lista com todas as features que vao utilizar mediana para os mv
median = ['PERSON_AGE', ]
# Lista com todas as features que vao utilizar media para os mv
medium = []
# Lista com todas as features que vao utilizar frequencia para os mv
most_frequent = [] 

tmp_nr, tmp_sb, tmp_bool = None, None, None
imp = SimpleImputer(strategy='median', missing_values=nan, copy=True)
tmp_nr = DataFrame(imp.fit_transform(data[median]), columns=median)

imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)

imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
df.to_csv(f'../../Projeto/data/NYC_collisions_tabular_mv_drop_fill.csv', index=False)
df.describe(include='all')


# --------------------- SECOND DATASET ------------------------------
register_matplotlib_converters()
filepath = '../../data/secondDataset/air_quality_tabular.csv'
data = read_csv(filepath, index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)

# Cria um grafico com o numero de missing values por variavel
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

# Remove as variaveis que têm 80% ou mais de missing values
# Remove tambem umas consideradas desnecessárias
threshold = data.shape[0] * 0.80
drops = [c for c in mv.keys() if mv[c] > threshold]
drops.append('date')
drops.append('City_EN')
drops.append('Prov_EN')
df = data.drop(columns=drops, inplace=False)

# Filling aproaches:
tmp_median, tmp_mean, tmp_sb, tmp_bool = None, None, None, None
variables = get_variable_types(df)
numeric_vars_median = ['CO_Mean', 'CO_Min',
                       'NO2_Mean', 'NO2_Min', 'NO2_Max', 'NO2_Std',
                       'O3_Mean', 'O3_Min', 'O3_Max', 'O3_Std',
                       'SO2_Min']
numeric_vars_mean = ['Field_1', 'CO_Max', 'CO_Std',
                     'PM2.5_Mean', 'PM2.5_Min', 'PM2.5_Max', 'PM2.5_Std',
                     'PM10_Mean', 'PM10_Min', 'PM10_Max', 'PM10_Std',
                     'SO2_Mean', 'SO2_Max', 'SO2_Std']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

# Cria um novo csv file com mean para numerics e most_frequent para symbolics e binaries
tmp_nr, tmp_sb, tmp_bool = None, None, None
if len(numeric_vars_median) > 0:
    imp = SimpleImputer(strategy='median', missing_values=nan, copy=True)
    tmp_median = DataFrame(imp.fit_transform(data[numeric_vars_median]), columns=numeric_vars_median)
if len(numeric_vars_mean) > 0:
    imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
    tmp_mean = DataFrame(imp.fit_transform(data[numeric_vars_mean]), columns=numeric_vars_mean)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

df = concat([tmp_median, tmp_mean, tmp_sb, tmp_bool], axis=1)
df.to_csv(f'../../Projeto/data/air_quality_tabular_mv_drop_fill.csv', index=False)
df.describe(include='all')
