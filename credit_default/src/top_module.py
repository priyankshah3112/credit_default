import pandas as pd
import numpy as np
import os
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config
from copy import deepcopy
from src.feature_engineering import chi_squared_score,feature_importance_extraTrees,mutual_info,pca_decomposition
from src.data_cleaning_and_preprocessing import convert_string_features,drop_nan_columns,\
    drop_targetY_nulls,drop_columns,fill_missing_values,create_dummy_variables,check_categorical_columns,\
    standardize_continuous_data,scale_categorical_columns,z_score_outlier,outlier_standard_deviation,discretize_indicatiors
from src.data_stats import feature_correlation,feature_statistics,targetY_count
from src.data_visualization import histograms_plot,box_plot,correlation_matrix_plot,scatter_matrix_plot
from src.machine_learning import compare_ml_algo,grid_search,final_model,validate_out_of_sample,prediction
from pickle import dump
from pickle import load

# function to draw different plots for the features in the data
def visualize_data(df_train,label='unkown'):

    histograms_plot(df_train,label)
    box_plot(df_train)
    correlation_matrix_plot(df_train,label)

# function to rank the features in accordance to it's relevance to targetY
def feature_engineering(df_train,label="unkown"):

    print('Ranking features according to chi_squared test')
    chi_squared_score(df_train,label)
    print("Chi Square scores csv generated")
    print('Ranking features according to importance given by ExtraTrees')
    feature_importance_extraTrees(df_train,label)
    print("feature importance given by ExtraTree stored in csv file")
    mutual_info(df_train,label)

# function to generate initial stats for raw data
def generate_data_stats(df_original):

    df=deepcopy(df_original)
    print('Generating data stats only for numeric features')
    feature_statistics(df)
    feature_correlation(df)
    targetY_count(df)

# function to handle end to end data cleaning of training data
def clean_dataset(training_df):

    train_dataset = drop_nan_columns(training_df, percent_threshold=config.drop_column_threshold)
    print("coulmns having nulls values more that the defined threshold are dropped ")
    train_dataset=drop_columns(train_dataset,column_names=config.drop_columns)
    print("columns which user has defined have been dropped. Please refer config file : ",config.drop_columns)
    train_dataset=fill_missing_values(train_dataset)
    print('Missing values in all columns have been filled/eliminated with predefined strategy')
    train_dataset = create_dummy_variables(train_dataset, nominal_columns=config.nominal_columns)
    print('Nominal categorical values in all columns have been converted to dummy values')
    train_dataset, feature_mapper = convert_string_features(train_dataset)
    filename = 'cleaning_data_parameters/feature_mapper.sav'
    dump(feature_mapper, open(filename, 'wb'))
    print('None numeric data converted to numeric data by a simple one-to-one mapping function')
    categorical_cols = check_categorical_columns(train_dataset)
    non_calc_columns = [config.primary_key] + [config.target_y_col]
    continuous_cols = train_dataset.columns.values[np.in1d(train_dataset.columns.values,
                                                         categorical_cols + non_calc_columns, invert=True)]
    train_dataset=outlier_standard_deviation(train_dataset,continuous_cols)
    train_dataset.to_csv('clean_data/cleaned_training_data.csv',index=False)
    print("cleaned csv files are generated")

# function which creates an out of sample data set from the training set keeping the proportion of class frequency same
def create_out_of_sample_set(df_original):

    df=deepcopy(df_original)
    print("Dropping rows with null values of TargetY: ",config.target_y_col)
    df.dropna(subset=[config.target_y_col],inplace=True)
    percent_split=config.out_of_sample_split
    df_true_class=df[df['default']==True]
    true_class_out_of_sample=df_true_class.sample(frac=percent_split, random_state=200)
    true_class_training=df_true_class.drop(true_class_out_of_sample.index)
    true_class_out_of_sample.reset_index(drop=True,inplace=True)
    true_class_training.reset_index(drop=True,inplace=True)
    df_false_class = df[df['default'] == False]
    false_class_out_of_sample=df_false_class.sample(frac=percent_split, random_state=200)
    false_class_training=df_false_class.drop(false_class_out_of_sample.index)
    false_class_out_of_sample.reset_index(drop=True, inplace=True)
    false_class_training.reset_index(drop=True, inplace=True)
    out_of_sample_test=pd.concat((true_class_out_of_sample,false_class_out_of_sample))
    out_of_sample_test.reset_index(drop=True, inplace=True)
    out_of_sample_test.to_csv('raw_data/out_of_sample.csv',index=False)
    new_training_data=pd.concat((true_class_training,false_class_training))
    new_training_data.reset_index(drop=True, inplace=True)
    new_training_data.to_csv('raw_data/new_training_data.csv',index=False)
    print("out of sample and new training dataset generated")

# funnction which handles the standardization and scaling of data
def standardize_dataset(training_df):

    categorical_cols=check_categorical_columns(training_df)
    non_calc_columns=[config.primary_key]+[config.target_y_col]
    continuous_cols=training_df.columns.values[np.in1d(training_df.columns.values,
                                                       categorical_cols+non_calc_columns,invert=True)]
    training_df,scaler=standardize_continuous_data(training_df,continuous_cols)
    filename = 'cleaning_data_parameters/feature_scaler.sav'
    dump(scaler, open(filename, 'wb'))
    filename_2='cleaning_data_parameters/columns_scaled.sav'
    dump(continuous_cols, open(filename_2, 'wb'))
    training_df=scale_categorical_columns(training_df,categorical_cols)
    training_df.to_csv('clean_standardized_data/clean_standardized_training_data.csv',index=False)

# function to preprocess any new data after ML model is created (used for test and out of samaple data)
def preprocess_new_data(df,label='unknown'):

    df_test=deepcopy(df)
    df_test=fill_missing_values(df_test)
    df_test=create_dummy_variables(df_test, nominal_columns=config.nominal_columns)
    if config.target_y_col in df.columns.values:
        features_needed = [config.primary_key]+[config.target_y_col] + config.selected_features
    else:
        features_needed = [config.primary_key] + config.selected_features
    filename = 'cleaning_data_parameters/feature_mapper.sav'
    feature_mapper=load(open(filename, 'rb'))
    df_test=convert_string_features(df_test, feature_mapper)
    df_test.to_csv('clean_data/'+str(label)+'_data.csv',index=False)
    filename = 'cleaning_data_parameters/feature_scaler.sav'
    scaler = load(open(filename, 'rb'))
    filename_2 = 'cleaning_data_parameters/columns_scaled.sav'
    continuous_cols=load(open(filename_2, 'rb'))
    categorical_cols = check_categorical_columns(df_test)
    df_test[continuous_cols] = scaler.transform(df_test[continuous_cols])
    df_test = df_test[features_needed]
    df_test = scale_categorical_columns(df_test, categorical_cols)
    df_test.to_csv('clean_standardized_data/clean_standardized_'+str(label)+'_data.csv',index=False)
    return df_test

# start the program from this function
def main():

    # ================================= Creation of out of sample test data ===========================================

    dataset = pd.read_csv('raw_data/puzzle_train_dataset.csv')
    answer=input("Do you want to create a out of sample test data from the raw data? (y/n)")
    if answer.lower()=="y":
        create_out_of_sample_set(dataset)
    del dataset

    # ================================== Generate statistics of raw data ==============================================

    answer=input("Do you want to generate the raw data statistics for new training data? (y/n)")
    if answer.lower()=="y":
        try:
            new_training_data = pd.read_csv('raw_data/new_training_data.csv')
        except:
            print("Please create an out of sample test data and new training data")
        generate_data_stats(new_training_data)

    # ========================== Cleaning of only new training data==================================================

    answer=input("Do you want to clean the raw data - only new_training_data? (y/n)")
    if answer.lower()=="y":
        try:
            new_training_data=pd.read_csv('raw_data/new_training_data.csv')
        except:
            print("Please create an out of sample test data and new training data")
        clean_dataset(new_training_data)

    # ========================== Standardizing of only new training data =======================

    answer=input("Do you want to standardize the cleaned_out_of_sample data?(y/n)")
    if answer.lower() == "y":
        try:
            cleaned_training_data=pd.read_csv('clean_data/cleaned_training_data.csv')
        except:
            print("Please clean out of sample test data and new training data")
        standardize_dataset(cleaned_training_data)

    # ========================== Visualization of variables of training data ===========================================

    answer = input("Do you want to visualize the clean data?(y/n)")
    if answer.lower() == "y":
        try:
            cleaned_training_data = pd.read_csv('clean_standardized_data/clean_standardized_training_data.csv')
        except:
            print("Standardized clean data file is not present")
        visualize_data(cleaned_training_data,label='_clean_standardized')

    # ========================== Feature Engineering on training data ================================================

    answer = input("Do you want to check importance of individual feature of clean data?(y/n)")
    if answer.lower() == "y":
        try:
            cleaned_training_data = pd.read_csv('clean_standardized_data/clean_standardized_training_data.csv')
        except:
            print("Standardized clean data file is not present")
        feature_engineering(cleaned_training_data,label='_full_data')

    # ============================ PCA decomposition to see how much variance is shown ================================

    answer = input("Do you want to do PCA decompose of clean_standardized_data to check varinace?(y/n)")
    if answer.lower() == "y":
        try:
            cleaned_training_data = pd.read_csv('clean_standardized_data/clean_standardized_training_data.csv')
        except:
            print("Standardized clean data file is not present")
        pca_decomposition(cleaned_training_data,n_components=config.pca_components_output)
    # ============================================ ML model comparision =============================================

    answer = input("Do you want to compare various ML models on clean_standardized_data for selected features?(y/n)")
    if answer.lower() == "y":
        try:
            training_df =pd.read_csv('clean_standardized_data/clean_standardized_training_data.csv')
        except:
            print("Standardized clean data file is not present")
        selected_features=config.selected_features
        compare_ml_algo(training_df,selected_features=selected_features,scoring='roc_auc')
    # ============================ Run Grid search on selected ML model ===============================================

    answer = input("Do you want to run grid search for the selected ML model?(y/n)")
    if answer.lower() == "y":
        try:
            training_df = pd.read_csv('clean_standardized_data/clean_standardized_training_data.csv')
        except:
            print("Standardized clean data file is not present")
        selected_features = config.selected_features
        grid_search(training_df,selected_features=selected_features,scoring='roc_auc')

    # ============================ Create the final ML model ===========================================================

    answer = input("Do you want to create the final ML model?(y/n)")
    if answer.lower() == "y":
        try:
            training_df = pd.read_csv('clean_standardized_data/clean_standardized_training_data.csv')
        except:
            print("Standardized clean data file is not present")
        selected_features = config.selected_features
        final_model(training_df, selected_features=selected_features, scoring='roc_auc')

    # ============================ Validate Trained ML model on out of sample data =====================================

    answer = input("Do you want to check the trained ML model accuracy on out of sample data?(y/n)")
    if answer.lower() == "y":
        try:
            out_sample_data = pd.read_csv('raw_data/out_of_sample.csv')
        except:
            print("Standardized clean out of sample data file is not present")
        new_df=preprocess_new_data(out_sample_data,label='out_of_sample')
        selected_features = config.selected_features
        validate_out_of_sample(new_df, selected_features=selected_features)

    # ============================ Predictions for puzzle_test_data =====================================

    answer = input("Do you want to test ML model on unseen data (puzzle_test_dataset)?(y/n)")
    if answer.lower() == "y":
        try:
            test_data = pd.read_csv('raw_data/puzzle_test_dataset.csv')
        except:
            print("test data file is not present")
        test_data=preprocess_new_data(test_data,label='puzzle_test')
        prediction(test_data)



if __name__ == '__main__':
    main()
