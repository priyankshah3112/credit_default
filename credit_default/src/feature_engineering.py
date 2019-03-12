from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
import os
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config

# function to generate random features for setting a bench mark for other features to beat
def random_indicator(df_original):
    df = deepcopy(df_original)
    for i in range(0, config.no_rand):
        df['random_variable_' + str(i + 1)] = np.random.normal(size=df.shape[0])
    return df

# function to generate the chi squared scores to rank features
def chi_squared_score(df_original,label):

    df=deepcopy(df_original)
    df = random_indicator(df)
    for col in df.columns.values:
        if col==config.primary_key:
            continue
        if df[col].min()<0:
            df[col]=df[col]+abs(df[col].min())
    Y=df[config.target_y_col]
    input_features=df.columns.values[np.in1d(df.columns.values,[Y.name]+[config.primary_key],invert=True)].tolist()
    X=df[input_features]
    test = SelectKBest(score_func=chi2, k=10)
    fit = test.fit(X, Y)
    scores=fit.scores_
    score_df=pd.DataFrame(X.columns.values,columns=["feature_name"])
    score_df['scores']=scores
    score_df.sort_values('scores',ascending=False,inplace=True)
    score_df.reset_index(drop=True,inplace=True)
    score_df.to_csv('reports/chi_squared'+str(label)+'.csv',index=False)
    return score_df

# function to generate the rank of features using extaTrees - wrapper method
def feature_importance_extraTrees(df_original,label):
    df = deepcopy(df_original)
    Y = df[config.target_y_col]
    input_features = df.columns.values[np.in1d(df.columns.values, [Y.name] + [config.primary_key], invert=True)].tolist()
    X = df[input_features]
    X = random_indicator(X)
    model = ExtraTreesClassifier()
    model.scoring='auc_roc'
    model.fit(X, Y)
    scores=model.feature_importances_
    score_df = pd.DataFrame(X.columns.values, columns=["feature_name"])
    score_df['scores'] = scores
    score_df.sort_values('scores', ascending=False, inplace=True)
    score_df.reset_index(drop=True, inplace=True)
    score_df.to_csv('reports/feature_importance_extraTrees'+str(label)+'.csv', index=False)
    return score_df

# function to generate the MI scores to rank features
def mutual_info(df_original,label):

    df = deepcopy(df_original)
    Y = df[config.target_y_col]
    input_features = df.columns.values[
        np.in1d(df.columns.values, [Y.name] + [config.primary_key], invert=True)].tolist()
    X = df[input_features]
    X = random_indicator(X)
    test = SelectKBest(score_func=mutual_info_classif, k=5)
    fit = test.fit(X, Y)
    scores = fit.scores_
    print(X.columns.values)
    print(scores)
    score_df = pd.DataFrame(X.columns.values, columns=["feature_name"])
    score_df['scores'] = scores
    score_df.sort_values('scores', ascending=False, inplace=True)
    score_df.reset_index(drop=True, inplace=True)
    score_df.to_csv('reports/mutual_info'+str(label)+'.csv', index=False)
    return score_df

# function to decompose the input features and check the variance observed
def pca_decomposition(df_original,n_components):

    non_calc_columns = [config.primary_key] + [config.target_y_col]
    columns_to_decompose=df_original.columns.values[[np.in1d(df_original.columns.values,non_calc_columns,invert=True)]]
    df = deepcopy(df_original)
    X = df[columns_to_decompose]
    pca = PCA(n_components=n_components)
    fit=pca.fit(X)
    print("Explained Variance:", fit.explained_variance_ratio_)
    return

