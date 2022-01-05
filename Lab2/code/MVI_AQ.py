# ============================ #
# -------SECOND DATASET------- #
# ============================ #

import pandas as pd
import numpy as np
from numpy import nan
from ds_charts import get_variable_types, bar_chart
from pandas import concat, DataFrame, read_csv
from sklearn.impute import SimpleImputer
from matplotlib.pyplot import figure, savefig
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
filepath = '../../data/secondDataset/air_quality_tabular.csv'
data = read_csv(filepath, index_col=['date', 'FID'], na_values='', parse_dates=True, infer_datetime_format=True)
df = pd.DataFrame(data)

print("----------------------DV------------------------")
# counts the number of different values in each column(ids have a very large number)
for c in df.columns:
    print(c, len(df[c].dropna(inplace=False).unique()))

print("----------------------MV------------------------")
# gets missing values from each column
for c in df.columns:
   print(c, df[c].isna().sum())

# separate variable types
variables = get_variable_types(data)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

numeric_vars_mean = ['Field_1', 'CO_Max', 'CO_Std', 'PM2.5_Mean',
                     'PM2.5_Min', 'PM2.5_Max', 'PM2.5_Std', 'PM10_Mean',
                     'PM10_Min', 'PM10_Max', 'PM10_Std', 'SO2_Mean', 
                     'SO2_Max', 'SO2_Std']

numeric_vars_median = ['CO_Mean', 'CO_Min', 'NO2_Mean', 'NO2_Min', 'NO2_Max',
                       'NO2_Std', 'O3_Mean', 'O3_Min', 'O3_Max', 'O3_Std',
                       'SO2_Min']
symbolic_vars = []
binary_vars = []

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
df.to_csv(f'../../data/secondDataset/air_quality_tabular_no_mv.csv', index=False)
df.describe(include='all')