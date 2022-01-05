from numpy import number
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ds_charts import get_variable_types
from pandas import DataFrame, concat, read_csv
from pandas.plotting import register_matplotlib_converters
import pandas as pd

register_matplotlib_converters()
file = "air_quality"
filename = '../../data/firstDataset/NYC_collisions_tabular_no_mvs.csv'
data = read_csv(filename, na_values='', parse_dates=True, infer_datetime_format=True)

# Drop out all records with missing values
# should be 0
data.dropna(inplace=True)

def dummify_OneHot(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=bool)
    X = df[vars_to_dummify]
    encoder.fit(X)
    new_vars = encoder.get_feature_names(vars_to_dummify)
    trans_X = encoder.transform(X)
    dummy = DataFrame(trans_X, columns=new_vars, index=X.index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

# dummifies all symbolic variables
variables = get_variable_types(data)
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']
df = dummify_OneHot(data, binary_vars)
df.to_csv(f'../../data/firstDataset/NYC_collisions_tabular_dummified.csv', index=False)
df.describe(include=[bool])
