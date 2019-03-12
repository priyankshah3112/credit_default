from matplotlib import pyplot as plt
import numpy as np
from pandas.tools.plotting import scatter_matrix
import os
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(src_dir)
from config import config

def histograms_plot(df,label=''):
    df.hist()
    plt.show()
    plt.savefig('graphs/feature_histograms'+str(label)+'.png')
    return

def box_plot(df,label=''):
    for col in df.columns.values:
        if col==config.primary_key:
            continue
        df[col].plot(kind='box')
        plt.show()
        plt.savefig('graphs/feature_boxplot' + str(col) +str(label)+ '.png')
    return

def correlation_matrix_plot(df,label=''):
    df.drop('ids',axis=1,inplace=True,errors='ignore')
    names=df.columns.values.tolist()
    correlations = df.corr()
    correlations.to_csv('reports/new_features_correlation.csv')
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 26, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names,rotation=90)
    ax.set_yticklabels(names)
    plt.show()
    plt.savefig('graphs/feature_correlation_plot'+str(label)+'.png')

def scatter_matrix_plot(df,label=''):
    scatter_matrix(df)
    plt.show()
    plt.savefig('graphs/scatter_matrix_plot'+str(label)+'.png')
