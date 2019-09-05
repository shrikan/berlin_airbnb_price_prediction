import pandas as pd
import importlib.util
from mlmodels import xg_boost, kernel_ridge_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#script_path = "../model/kernel_ridge_regression.py"
#spec = importlib.util.spec_from_file_location("execute", script_path)
#module = importlib.util.module_from_spec(spec)
#spec.loader.exec_module(module)


"""
    @param df : the dataframe having few rows with null area value
    
    :returns the dataframe which is filled with area values from regression
    
    The missing values are obtained from regression.
    Kernel ridge regression is used for the same with RBF kernel.
    The data is reduced to only few columns which kind of are dependant on area

"""


def get_missing_values(df):
    reduced_df = df[['bedrooms', 'price', 'cleaning_fee', 'distance', 'extra_people', 'area']]
    df.drop(['bedrooms', 'price', 'cleaning_fee', 'distance', 'extra_people', 'area'], axis=1, inplace=True)

    training_data = reduced_df[reduced_df['area'].notnull()]
    test_data = reduced_df[reduced_df['area'].isnull()]

    # Training req
    X_train = training_data.drop(['area'], axis=1)
    y_train = training_data['area']

    # Testing req
    X_test = test_data.drop(['area'], axis=1)

    y_test = kernel_ridge_regression.execute(X_train, y_train, X_test)

    y_test = pd.DataFrame(y_test)
    y_test.columns = ['area']

    # Indices must be handled properly while merging otherwise creates some NaN entries
    pre_index = pd.DataFrame(X_test.index)
    pre_index.columns = ['pre_index']

    y_test = pd.concat([y_test, pre_index], axis=1)
    y_test.set_index(['pre_index'], inplace=True)

    merged_data = pd.concat([X_test, y_test], axis=1)

    reduced_df = pd.concat([merged_data, training_data], axis=0)
    merged_df = pd.concat([reduced_df, df], axis=1)

    return merged_df


def get_price(df):
    # Set the target column and remove it from the dataframe
    target = df[['price']]
    df.drop(['price'], axis=1, inplace=True)

    xg_boost.execute(df, target)
