import pandas as pd
import numpy as np
from numpy import nan
from ds_charts import get_variable_types, bar_chart
from pandas import concat, DataFrame, read_csv
from sklearn.impute import SimpleImputer
from matplotlib.pyplot import figure, savefig
from pandas.plotting import register_matplotlib_converters


# FIRST DATASET
register_matplotlib_converters()
filename = '../../data/firstDataset/NYC_collisions_tabular.csv'
data = read_csv(filename, index_col='UNIQUE_ID', na_values='', parse_dates=True, infer_datetime_format=True)
df = pd.DataFrame(data)

# Cria um grafico com o numero de missing values por variavel
mv = {}
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

# separate variable types
variables = get_variable_types(data)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

# Person Age uses median 
df.loc[df['PERSON_AGE'].isna(), 'PERSON_AGE'] = df['PERSON_AGE'].dropna(inplace=False).median()

# Gets most frequent value for each BODILY_INJURY (it's always lap Belt & Harness except for Knee-Lower Leg Foot ---> None)
# fills mvs
df.loc[df['SAFETY_EQUIPMENT'].isna() & df['BODILY_INJURY'] == "Knee-Lower Leg Foot", 'SAFETY_EQUIPMENT'] = "None"
df.loc[df['SAFETY_EQUIPMENT'].isna(), 'SAFETY_EQUIPMENT'] = "Lap Belt & Harness"

# PERSON_SEX and PERSON_TYPE have no mvs
nan_constant_fill = "miss"
# PED_LOCATION and POSITION_IN_VEHICLE mvs fill
# for PED_LOCATION we only have to fill mvs with a constant, because every mv is from when PERSON_TYPE!= Pedestrain
df.loc[df['PED_LOCATION'].isna(), 'PED_LOCATION'] = nan_constant_fill

#For POSITION_IN_VEHICLE the same thing, but the constant symbolizes a Pedestrian
df.loc[df['POSITION_IN_VEHICLE'].isna() & df['PERSON_TYPE'] == "Occupant", 'POSITION_IN_VEHICLE'] = df['POSITION_IN_VEHICLE'].dropna(inplace=False).mode()
df.loc[df['PERSON_TYPE'] != "Occupant", 'POSITION_IN_VEHICLE'] = nan_constant_fill
df.loc[df['POSITION_IN_VEHICLE'].isna(), 'POSITION_IN_VEHICLE'] = nan_constant_fill

# The same goes for EJECTION na == Pedestrian
df.loc[df['PERSON_TYPE'] == 'Pedestrian', 'EJECTION'] = nan_constant_fill
df.loc[df['EJECTION'].isna(), 'EJECTION'] = df['EJECTION'].dropna(inplace=False).mode()
df.loc[df['EJECTION'].isna(), 'EJECTION'] = nan_constant_fill

#Again for VEHICLE_ID
df.loc[df['VEHICLE_ID'].isna() & df['PERSON_TYPE'] == "Occupant", 'VEHICLE_ID'] = df['VEHICLE_ID'].dropna(inplace=False).mode()
df.loc[df['VEHICLE_ID'].isna(), 'VEHICLE_ID'] = nan_constant_fill

#PED_ACTION == nan when PERSON_TYPE != Pedestrian
df.loc[df['PED_ACTION'].isna(), 'PED_ACTION'] = nan_constant_fill

# x y, Nothing to do.
# x na or na x = x x
# na na = cf1.mode cf2.mode
df.loc[df['CONTRIBUTING_FACTOR_1'].isna()] = 'Unspecified'
df.loc[df['CONTRIBUTING_FACTOR_2'].isna()] = 'Unspecified'

# Numero de missing values por variavel
mv = {}
for var in df:
    nr = df[var].isna().sum()
    if nr > 0:
        mv[var] = nr
print("missing values end", mv)

df.to_csv(f'../../data/firstDataset/NYC_collisions_tabular_no_mvs.csv', index=False)