
# function to generate the basic stats of data like min, max, 25 percentile, 75 percentile, etc.
def feature_statistics(df):
    description=df.describe()
    description.to_csv("reports/feature_stats.csv")
    print('feature statistics csv file generated')
    return

# function to generate the target Y class count
def targetY_count(df):
    targetY_count=df.groupby('default').size()
    targetY_count.to_csv("reports/target_Y_count.csv")
    print('targetY count csv file generated')
    return

# function to generate the correlation amongst features and also the target Y
def feature_correlation(df):
    correlations=df.corr(method='pearson')
    correlations.to_csv("reports/feature_correlations.csv")
    print('feature correlation csv file generated')
    return
