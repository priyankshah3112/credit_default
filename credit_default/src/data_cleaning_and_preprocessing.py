import pandas as pd
import numpy as np
from copy import deepcopy
import os
from scipy import stats
from sklearn.preprocessing import StandardScaler
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config

# function to convert non-numeric data to numeric by a simple mapper
def convert_string_features(df_original,main_mapper=None):
    df = deepcopy(df_original)
    # when new_training_data is passed, main_mapper is empty
    if main_mapper==None:

        main_mapper=dict()
        for col in df.columns.values:
            # if col is 'ids' then do not convert to numeric
            if col == config.primary_key:
                continue
            # default column should be converted to binary from boolean
            elif col == config.target_y_col:
                df[col]=df[col].astype(int)
            # if column is non-numeric then each unique value is mapped to a new number
            elif df[col].dtype=='O':
                mapper=dict()
                for i,name in enumerate(df[col].unique()):
                    try:
                        if not np.isnan(name):
                            mapper[name]=i
                    except:
                        mapper[name]=i

                df[col]=df[col].map(mapper)
                main_mapper[col]=mapper
        return df,main_mapper
    # when out_of_sample_data of test data is passed, main_mapper made from the training data is used
    else:
        for col in df.columns.values:
            if col == config.target_y_col:
                df[col]=df[col].astype(int)
            elif col in main_mapper.keys():
                df[col] = df[col].map(main_mapper[col])
        return df

# function to drop the columns which has null values greater than the threshold
def drop_nan_columns(df_original,percent_threshold):

    df=deepcopy(df_original)
    for col in df.columns.values.tolist():
        if len(df[df[col].isnull()]) / df.shape[0] * 100 > percent_threshold:
            df.drop(col,axis=1,inplace=True)

    return df

# function to drop the rows in which targetY is NULL
def drop_targetY_nulls(df_original,target_y_col="default"):

    df=deepcopy(df_original)
    df=df[~df[target_y_col].isnull()]
    return df

# funcrion to standardize data
def standardize_continuous_data(df_original,continuous_columns,scaler=None):

    # When standardizing the data for first time
    if scaler==None:
        df=deepcopy(df_original)
        df=df[continuous_columns]
        scaler = StandardScaler().fit(df)
        rescaledX = scaler.transform(df)
        df_original[continuous_columns]=rescaledX
        return df_original,scaler
    # When standardizing new data (Eg: out of sample data), hence using old scaler model
    else:
        df=deepcopy(df_original)
        df=df[continuous_columns]
        rescaledX = scaler.transform(df)
        df_original[continuous_columns] = rescaledX
        return df_original,scaler

# funcrion to drop specified columns
def drop_columns(df_original,column_names):

    df=deepcopy(df_original)
    df.drop(column_names,axis=1,inplace=True,errors='ignore')
    return df

# funcrion to fill missing values in the data - uses the strategy mentioned in the config file
def fill_missing_values(df_original,value_dict=None):

    df=deepcopy(df_original)
    if value_dict==None:
        value_dict=config.missing_values

    for col in value_dict.keys():
        if col in df.columns.values:
            if value_dict[col]=='drop_row':
                df.dropna(subset=[col],inplace=True)
            elif value_dict[col]=='drop_column':
                df.drop([col],axis=1,inplace=True)
            else:
                df[col].fillna(value_dict[col],inplace=True)

    return df

# funcrion to create a dummy variable for nominal categorical features Eg: gender is converted to gender_F and gender_M
def create_dummy_variables(df_original,nominal_columns=config.nominal_columns):

    df=deepcopy(df_original)
    for col in df.columns.values:
        if (df[col].dtype != 'O')&(col in nominal_columns):
            df[col]=df[col].astype('str')
    dummy_df = pd.get_dummies(df[nominal_columns])
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(nominal_columns, axis=1)
    return df

# function which returns the list of features which are categorical
def check_categorical_columns(df_original,categorical_threshold=config.categorical_thrshold):

    df=deepcopy(df_original)
    categorical_columns=[]
    non_calc_columns=[config.target_y_col]+[config.primary_key]
    df.drop(non_calc_columns,axis=1,inplace=True,errors='ignore')
    for col in df.columns.values:
        if len(df[col].unique())<=categorical_threshold:
            categorical_columns.append(col)

    return categorical_columns

# function which scales the categorical columns in the range of (0,1)
def scale_categorical_columns(df_original,categorical_cols):

    df=deepcopy(df_original)
    for col in categorical_cols:
        if col in df.columns.values:
            if len(df[col].unique())> 2:
                df[col]=df[col]/(len(df[col].unique())-1)
    return df

# function which removes outier with z>3 values
def z_score_outlier(df_original,continuous_cols):

    df=deepcopy(df_original)
    z = np.abs(stats.zscore(df[list(continuous_cols)]))
    new_df=df[(z < 3).all(axis=1)]
    return new_df

# function which corrects outlier based on std
def outlier_standard_deviation(df_original,continuous_cols):
    df=deepcopy(df_original)
    for col in continuous_cols:
        df[col]=np.where(df[col]>df[col].mean()+2*df[col].std(),df[col].mean()+2*df[col].std(),df[col])
        df[col] = np.where(df[col] < df[col].mean() - 2 * df[col].std(), df[col].mean() - 2 * df[col].std(), df[col])
    return df

# function used to create descrete bins for continuous indicators to remove outliers
def discretize_indicatiors(df_original,cols_to_disc):
    df=deepcopy(df_original)
    for col in cols_to_disc:
        df[col]=pd.qcut(df[col],q=config.disc_bins,labels=False)
    return df