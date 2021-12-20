from numpy import nan
from ds_charts import get_variable_types, bar_chart
from pandas import concat, DataFrame, read_csv
from sklearn.impute import SimpleImputer
from matplotlib.pyplot import figure, savefig
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()
filepath = '../../data/secondDataset/air_quality_tabular.csv'
data = read_csv(filepath, index_col='FID', na_values='', parse_dates=True, infer_datetime_format=True)


# Cria um grafico com o numero de missing values por variavel
mv = {}
figure()
for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr
        print('var:', var, 'mv:', nr)

bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
          xlabel='variables', ylabel='nr missing values', rotation=True)
savefig(f'../data/images/air_quality_tabular_missing_values.png')


# Dropping 1
# Ignora as variaveis que têm 90% ou mais de missing values
threshold = data.shape[0] * 0.90
missings = [c for c in mv.keys() if mv[c] > threshold]
df = data.drop(columns=missings, inplace=False)
df.to_csv(f'../data/air_quality_tabular_drop_columns_mv.csv', index=False)
print('Dropped variables', missings)


# Dropping 2
# Ignorar os records (linhas) que tenham 50% ou mais de missing values
threshold = data.shape[1] * 0.50
df = data.dropna(thresh=threshold, inplace=False)
df.to_csv(f'../data/air_quality_tabular_drop_records_mv.csv', index=False)
print(df.shape)


# Filling 1
# Filling aproaches:
#   constant: the constant value chosen depends on the type of variable (usually: NaN, -1 or 0 for numeric, 'NA' for symbolic and False for boolean
#   mean, only applicable for numeric variables
#   median, only applicable for numeric variables
#   most_frequentmostly applicable for symbolic variables
tmp_nr, tmp_sb, tmp_bool = None, None, None
variables = get_variable_types(data)
numeric_vars = variables['Numeric']
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']

if len(numeric_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value=0, missing_values=nan, copy=True)
    tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value='NA', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='constant', fill_value=False, missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)
# Cria um novo csv file com os missing values em NA
df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
df.to_csv(f'../data/air_quality_tabular_mv_constant.csv', index=False)
df.describe(include='all')


# Cria um novo csv file com mean para numerics e most_frequent para symbolics e binaries
tmp_nr, tmp_sb, tmp_bool = None, None, None
if len(numeric_vars) > 0:
    imp = SimpleImputer(strategy='mean', missing_values=nan, copy=True)
    tmp_nr = DataFrame(imp.fit_transform(data[numeric_vars]), columns=numeric_vars)
if len(symbolic_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_sb = DataFrame(imp.fit_transform(data[symbolic_vars]), columns=symbolic_vars)
if len(binary_vars) > 0:
    imp = SimpleImputer(strategy='most_frequent', missing_values=nan, copy=True)
    tmp_bool = DataFrame(imp.fit_transform(data[binary_vars]), columns=binary_vars)

df = concat([tmp_nr, tmp_sb, tmp_bool], axis=1)
df.to_csv(f'../data/air_quality_tabular_mv_most_frequent.csv', index=False)
df.describe(include='all')


