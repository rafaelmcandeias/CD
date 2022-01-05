from numpy import number
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ds_charts import get_variable_types
from pandas import DataFrame, concat, read_csv
from pandas.plotting import register_matplotlib_converters
import pandas as pd

register_matplotlib_converters()
file = "air_quality"
filename = '../../data/secondDataset/air_quality_tabular_no_mvs.csv'
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

def dummify_Label(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
        
    encoder = LabelEncoder()
    encoder.fit(df[vars_to_dummify[0]])
    trans = encoder.transform(df[vars_to_dummify[0]])
    dummy = DataFrame(trans, columns=['City_EN'], index=df[vars_to_dummify[0]].index)
    dummy = dummy.convert_dtypes(convert_boolean=True)

    encoder = LabelEncoder()
    encoder.fit(df[vars_to_dummify[1]])
    trans = encoder.transform(df[vars_to_dummify[1]])
    dummy2 = DataFrame(trans, columns=['Prov_EN'], index=df[vars_to_dummify[1]].index)
    dummy2 = dummy2.convert_dtypes(convert_boolean=True)

    final_df = concat([df[other_vars], dummy, dummy2], axis = 1)
    
    return final_df

# dummifies all symbolic variables
variables = get_variable_types(data)
symbolic_vars = variables['Symbolic']
binary_vars = variables['Binary']
df = dummify_OneHot(data, binary_vars)
df = dummify_Label(df, symbolic_vars)
df.to_csv(f'../../data/secondDataset/air_quality_tabular_dummified.csv', index=False)
df.describe(include=[bool])
