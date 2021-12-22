from numpy import nan
from ds_charts import get_variable_types, bar_chart
from pandas import concat, DataFrame, read_csv
from sklearn.impute import SimpleImputer
from matplotlib.pyplot import figure, savefig
from pandas.plotting import register_matplotlib_converters


# FIRST DATASET
register_matplotlib_converters()
filepath = '../../data/firstDataset/NYC_collisions_tabular.csv'
data = read_csv(filepath, index_col='UNIQUE_ID', na_values='', parse_dates=True, infer_datetime_format=True)

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
drops.append('CRASH_DATE')
drops.append('CRASH_TIME')
drops.append('COLLISION_ID')
df = data.drop(columns=drops, inplace=False)

# Filling aproaches:
tmp_nr, tmp_sb, tmp_bool = None, None, None
variables = get_variable_types(df)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

# Cria um novo csv file com mean para numerics e most_frequent para symbolics e binaries
tmp_nr, tmp_sb, tmp_bool = None, None, None
if len(numeric_vars) > 0:
    imp = SimpleImputer(strategy='median', missing_values=nan, copy=True)
    tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
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
