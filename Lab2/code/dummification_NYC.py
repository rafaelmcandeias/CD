from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ds_charts import get_variable_types
from pandas import DataFrame, concat, read_csv
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
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


def dummify_Label(df, vars_to_dummify):
    other_vars = [c for c in df.columns if not c in vars_to_dummify]
    final_df = df[other_vars]
    
    for var in vars_to_dummify:
        encoder = LabelEncoder()
        encoder.fit(df[var])
        trans = encoder.transform(df[var])
        dummy = DataFrame(trans, columns=[var], index=df[var].index)
        dummy = dummy.convert_dtypes(convert_boolean=True)
        final_df = concat([final_df, dummy], axis=1)

    return final_df


# dummifies all symbolic variables
vars_to_one = ['PERSON_SEX', 'PERSON_TYPE', 'PED_LOCATION', 'EJECTION', 'PED_ROLE']
vars_to_label = ['SAFETY_EQUIPMENT', 'BODILY_INJURY','CONTRIBUTING_FACTOR_2', 'CONTRIBUTING_FACTOR_1',
                 'COMPLAINT', 'EMOTIONAL_STATUS', 'POSITION_IN_VEHICLE', 'PED_ACTION', 'PERSON_ID', 'PERSON_INJURY']
df = dummify_OneHot(data, vars_to_one)  
df = dummify_Label(df, vars_to_label)
df.to_csv(f'../../data/firstDataset/NYC_collisions_tabular_dummified.csv', index=False)
df.describe(include=[bool])
