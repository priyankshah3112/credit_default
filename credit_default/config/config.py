# threshold of null values for dropping a column
drop_column_threshold=20
# primary key of the dataset
primary_key='ids'
# targetY columns to be predicted
target_y_col='default'
# percentage of data kept for out of sample validation
out_of_sample_split=0.15
# User's descretion for dropping irrelevant columns (Eg: channel has only one value or reason has too many values which cannot be quantified)
drop_columns=['reason','job_name','n_issues','channel','zip','state']
# Value used to fill missing data for the respective columns
missing_values={'gender':'unknown','facebook_profile':'unknown','n_defaulted_loans':'drop_row','n_bankruptcies':'drop_row',
                'n_accounts':'drop_row','real_state':'drop_row','income':'drop_row','reason':'drop_row','risk_rate':'drop_row',
                'amount_borrowed':'drop_row','score_1':'drop_row','score_2':'drop_row','score_3':'drop_row','score_4':'drop_row',
                'score_5':'drop_row','score_6':'drop_row'}
# Nominal columns have categorical data which has no order. New dummy values have to be created for these
nominal_columns=['facebook_profile','gender','borrowed_in_months']
# categorical_thrshold: thrshold value to decide if a column is categorical or continuous
categorical_thrshold=14
# No of components of PCA to be considered for decomposition
pca_components_output=2
# selected feature list that will be used to train ML models
selected_features=['risk_rate','amount_borrowed','facebook_profile_True','facebook_profile_False','gender_f','borrowed_in_months_60.0','gender_m','income','score_3']
# columns to be discretized
cols_to_disc=['score_3','score_4','score_5','score_6','risk_rate','income','amount_borrowed','n_accounts']
# bins to be created for discretization
disc_bins=5
# ML model whose hyper parameter is to be tunned
ML_model='GradientBoostingClassifier()'
# Hyper parameters to be tunned
model_hyper_parameters="dict(learning_rate=[0.05,0.1,1.5,0.2],max_depth=[2,3,4,5],n_estimators =[40,80,150,200])"
# Final ML model - this model is stored as a pickle for use of testing
ML_model_tunned='GaussianNB()'
# ML_model_tunned='GradientBoostingClassifier(learning_rate=.05,max_depth=4,n_estimators=200)'
# no of random features to be generated into the dataset to check for feature relevance benchmark
no_rand=5