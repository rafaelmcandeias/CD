from numpy import nan
from ds_charts import get_variable_types
from pandas import concat, DataFrame, read_csv
from sklearn.impute import SimpleImputer
from pandas.plotting import register_matplotlib_converters


# PERSON_SEX and PERSON_TYPE have no mvs
nan_constant_fill = -1



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
print("missing values start", mv)

# PERSONG_AGE mvs are filled with median values
df.loc[df['PERSON_AGE'].isna(), 'PERSON_AGE'] = df['PERSON_AGE'].median()

# Gets most frequent BODILY_INJURY for each SAFETY_EQUIPMENT 
# (it's always BODILY_INJURY = lap Belt & Harness except 
# for SAFETY_EQUIPMENT = Knee-Lower Leg Foot, where BODILY_INJURY = None)x\
values = df.groupby('BODILY_INJURY')['SAFETY_EQUIPMENT'].apply(lambda x: x.value_counts().index[0]).reset_index()

# fills mvs
df.loc[df['SAFETY_EQUIPMENT'].isna() & df['BODILY_INJURY'] == "Knee-Lower Leg Foot", 'SAFETY_EQUIPMENT'] = "None"
df.loc[df['SAFETY_EQUIPMENT'].isna(), 'SAFETY_EQUIPMENT'] = "Lap Belt & Harness"

# For EJECTION, Pedestrians shouldn't have and most are missing. 
df.loc[df['PERSON_TYPE'] == "Pedestrian", 'EJECTION'] = nan_constant_fill
# Occupants tem poucos mv, usa se most_frequent
df.loc[df['EJECTION'].isna() & df['PERSON_TYPE'] == "Occupant", 'EJECTION'] = df['EJECTION'].mode()

# Puts all VEHICLE_ID in Pedestrian as nan since it doenst make sence to associate both
df.loc[df['VEHICLE_ID'].isna() & df['PERSON_TYPE'] == "Pedestrian" , 'VEHICLE_ID'] = nan_constant_fill
# Puts all VEHICLE_ID in Occupant as the value from another register with the same COLLISION_ID? 
# Only 9% isn't nan, even from another register. Not worth it. 
# Apply most_frequent instead
df.loc[df['VEHICLE_ID'].isna() & df['PERSON_TYPE'] == "Occupant", 'VEHICLE_ID'] = df['VEHICLE_ID'].mode()

#For POSITION_IN_VEHICLE the same thing
df.loc[df['PERSON_TYPE'] == "Pedestrian", 'POSITION_IN_VEHICLE'] = nan_constant_fill
df.loc[df['POSITION_IN_VEHICLE'].isna() & df['PERSON_TYPE'] == "Occupant", 'POSITION_IN_VEHICLE'] = df['POSITION_IN_VEHICLE'].mode()

# na = unspecified ou mv.
# Passar todos os mv para Unspecified. Facilita a condicao
cf1 = 'CONTRIBUTING_FACTOR_1'
cf2 = 'CONTRIBUTING_FACTOR_2'
df.loc[df[cf1].isna()] = 'Unspecified'
df.loc[df[cf2].isna()] = 'Unspecified'
# x y, Nothing to do.
# x na or na x = x x
for i in range(len(df[cf1].values)):
    if df[cf1].values[i] == 'Unspecified' and df[cf2].values[i] != 'Unspecified':
        df[cf1].values[i] = df[cf2].values[i]
    if df[cf1].values[i] != 'Unspecified' and df[cf2].values[i] == 'Unspecified':
        df[cf2].values[i] = df[cf1].values[i]
# na na = cf1.mode cf2.mode
df.loc[df[cf1] == 'Unspecified' & df[cf2] == 'Unspecified', cf1] = df[cf1].mode()
df.loc[df[cf1] == 'Unspecified' & df[cf2] == 'Unspecified', cf2] = df[cf2].mode()

# PED_LOCATION mvs fill
# for PED_LOCATION we only have to fill every value where PERSON_TYPE != Pedestrian. It's only for pedestrians
df.loc[df['PED_LOCATION'].isna() & df['PERSON_TYPE'] != 'Pedestrian', 'PED_LOCATION'] = nan_constant_fill

#PED_ACTION == nan when PERSON_TYPE == Pedestrian
df.loc[df['PED_ACTION'].isna(), 'PED_ACTION'] = nan_constant_fill

# Numero de missing values por variavel
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr
print("missing values end", mv)

# Saves dataframe in to csv file
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
