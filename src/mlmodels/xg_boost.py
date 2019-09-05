"""
XGBoost os an ensemble learning method in Machine learning.
Some of the major benefits of XGBoost are that its highly scalable/parallelizable,
quick to execute, and typically outperforms other algorithms.

source : https://towardsdatascience.com/exploring-xgboost-4baf9ace0cf6
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

"""
    @:param df Dataframe without the target column
            taregt Dataframe with target column

    :returns xgb_regressor A fit model with minimlastic RMSE to predict further values

    A random forest is used here to determine the best fit parameters and then xgboost
    method is used to find out the estimated price/target.
    A final plot is plotted in order to show relation with various features importance.
"""


def execute(df, target):
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2)

    # Scaling the data
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror')

    grid_parameters = {'n_estimators': [100, 150, 200],
                       'learning_rate': [0.01, 0.05, 0.1],
                       'gamma': [0.0, 0.1, 0.2],
                       'max_depth': [3, 4, 5, 6, 7],
                       'colsample_bytree': [0.6, 0.7, 1]
                       }

    random_forest_boost = GridSearchCV(xgb_regressor, grid_parameters, cv=3, n_jobs=-1)

    random_forest_boost.fit(X_train, y_train)

    # Now call the XGB Regression method with the best params obtained above
    best_params = random_forest_boost.best_params_

    xgb_regressor = xgb.XGBRegressor(**best_params, random_state=4)

    xgb_regressor.fit(X_train, y_train)

    # Prediction of values based on above trained model
    y_prediction = xgb_regressor.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
    print(f"RMSE by the used method is: {rmse}")

    # For visual better understanding, it is plotted with matplotlib
    feature_importances = pd.Series(xgb_regressor.feature_importances_, index=df.columns)
    feature_importances.nlargest(15).sort_values().plot(kind='barh', color='blue', figsize=(10, 5))
    plt.xlabel('Feature impact on the price')

    plt.show()

    return xgb_regressor
