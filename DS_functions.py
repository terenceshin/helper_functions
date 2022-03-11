import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_gbq as pgbq
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE 


###################
# DATA COLLECTION #
###################
def query_to_df(query):
    '''
    Input: query = query in the form of a string
    Output: DataFrame that contains output from query
    '''
    return pgbq.read_gbq(query, 'tensile-oarlock-191715', dialect='standard')


####################
# DATA EXPLORATION #
####################
def df_info(df):
    
    print(df.shape)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df.head())

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df.tail())
    
    x = pd.DataFrame(data={
    'missing_pct': df.isnull().sum() * 100 / len(df),
    'nunique': df.nunique(dropna=True), 
    'dtypes': df.dtypes,
    'sample1': df.sample(n=1, random_state=1).transpose().squeeze(),
    'sample2': df.sample(n=1, random_state=2).transpose().squeeze(),
    'sample3': df.sample(n=1, random_state=3).transpose().squeeze()
     }
    )

    x = x.join(df.describe(include='all', datetime_is_numeric=True).transpose())

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(x)
        
    return


def disc_central_tendency(df, column_name, bins):
    '''
    Input:  df = dataframe that you want to analyze
            column_name = specific column in df that you want to analyze
            bins = # of bins for histogram
    Output: histogram, mean, median, mode, standard deviation of NUMERICAL column
    '''
    fig = px.histogram(df, x=column_name, nbins=bins)
    fig.show()

    mean = df[column_name].mean()
    median = df[column_name].round(-1).median()
    mode = df[column_name].round(-1).mode()[0]
    st_dev = df[column_name].std()
    count = df[column_name].count()
    print("MEAN: " + str(mean))
    print("MEDIAN: " + str(median))
    print("MODE: " + str(mode))
    print("ST.DEV: " + str(st_dev))
    print("COUNT: " + str(count))


def cts_central_tendency(df, column_name):
    '''
    Input:  df = dataframe that you want to analyze
            column_name = specific column in df that you want to analyze
            bins = # of bins for histogram
    Output: KDE, mean, median, mode, standard deviation of NUMERICAL column
    '''
    df[column_name].plot.kde()

    mean = df[column_name].mean()
    median = df[column_name].round(-1).median()
    mode = df[column_name].round(-1).mode()[0]
    st_dev = df[column_name].std()
    count = df[column_name].count()
    print("MEAN: " + str(mean))
    print("MEDIAN: " + str(median))
    print("MODE: " + str(mode))
    print("ST.DEV: " + str(st_dev))
    print("COUNT: " + str(count))


def two_dim_density_func(df, x_col, y_col):

    figure = plt.figure(figsize = (12,12))
    sns.kdeplot(data = df, x = x_col, y = y_col, fill=True, cbar=True)
    plt.scatter(x = df[x_col], y = df[y_col])


####################
# DATA PREPARATION #
####################
def resample_data(X_train, y_train):
    '''
    Input:  X_train and y_train
    Output: Resampled X_train and y_train
    '''
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

###############################
# DATA MODELLING & EVALUATION #
###############################
def create_classification_model(df, X, y):
    '''
    Input:  df = DataFrame with X and y
            X = feature variables
            y = target variable
    Output: prints classification report of preliminary model, returns model and pred vs actual
    '''
    X = X.astype(float)
    y = y.astype(int)
    X = X.fillna(0)
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    
    # Joining results to df
    y_hats  = pd.DataFrame(y_pred).rename(columns={0:'predictions'})

    df_out = pd.merge(df, y_hats, how = 'inner',left_index = True, right_index = True)

    return model, df_out, X


def plot_feat_importance(model, X, size):
    """
    plots feature importance of a model
    
    Inputs: model (object): the model which its features will be plotted
            X (list): the list of features
            size (int): the number of features to plot 
        
    Output: Plot of feature importance for a given model and dataframe
    """  
    feat_importances = pd.Series(model.feature_importances_, index=list(X))
    feat_importances.nlargest(size).plot(kind='barh', figsize=(20,20))
    
    df = pd.DataFrame(model.feature_importances_, index=list(X)).sort_values(by=0, ascending=False)
    return df