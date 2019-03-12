from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from  sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import os
from pickle import dump
from pickle import load
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config

# function to compare different ML models and to plot the score metrics
def compare_ml_algo(df,selected_features,scoring):

    X=df[selected_features]
    Y=df[config.target_y_col]
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('GBC',GradientBoostingClassifier()))
    # each model is evaluated in turn
    results = []
    names = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=5, random_state=7)
        cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print(name, cv_results.mean(), cv_results.std())
    fig = pyplot.figure()
    fig.suptitle('ML model Comparison')
    ax = fig.add_subplot(111)
    pyplot.boxplot(results)
    ax.set_xticklabels(names)
    pyplot.show()
    pyplot.savefig('graphs/ml_comparision.png')
    return

# function to perform hyper parameters tunning on a particular ML model - currently made for GradientBoostingClassifier
def grid_search(df,selected_features,scoring):

    print('Grid search for best hyper parameters has started')
    param_grid=eval(config.model_hyper_parameters)
    X = df[selected_features]
    Y = df[config.target_y_col]
    model=eval(config.ML_model)
    kfold = StratifiedKFold(n_splits=5, random_state=7)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring=scoring, cv=kfold)
    grid.fit(X, Y)
    print(grid.best_score_)
    print(grid.best_estimator_.learning_rate,grid.best_estimator_.max_depth,grid.best_estimator_.n_estimators)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    return

# function to train the final ML model and to pickle it
def final_model(training_df, selected_features, scoring):

    X = training_df[selected_features]
    Y = training_df[config.target_y_col]
    model = eval(config.ML_model_tunned)
    model.scoring=scoring
    model.fit(X, Y)
    dump(model, open('saved_ML_model/finalized_model.sav', 'wb'))
    return

# function to return the confussion matrix and roc_auc of ML model on out_of_sample data
def validate_out_of_sample(out_sample, selected_features):

    X_test = out_sample[selected_features]
    Y_test = out_sample[config.target_y_col]
    loaded_model = load(open('saved_ML_model/finalized_model.sav', 'rb'))
    result = loaded_model.score(X_test, Y_test)
    print(loaded_model.scoring)
    print(result)
    Y_predicted=loaded_model.predict(X_test)
    accuracy=roc_auc_score(Y_test,Y_predicted)
    print(confusion_matrix(Y_test,Y_predicted))
    print(accuracy)
    return

# function to return the predictions of trained ML model on unseen data
def prediction(df_test):

    selected_features=config.selected_features
    loaded_model = load(open('saved_ML_model/finalized_model.sav', 'rb'))
    Y_prediction=loaded_model.predict(df_test[selected_features])
    df_test['default']=Y_prediction
    df_test['default']=df_test['default'].astype(bool)
    df_test=df_test[['ids','default']]
    df_test.to_csv('reports/predictions.csv',index=False)
    print('predictions csv file gennerated in reports folder')
    return df_test