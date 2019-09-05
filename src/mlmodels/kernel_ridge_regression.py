import pandas as pd
from sklearn.kernel_ridge import KernelRidge

"""
    @:param X_train Training input data 
            y_train Training input target
            X_test  Validation data
    
    :returns y_test Target values for the validation data
    
    A ridge regression with rbf kernel is used here with alpha value 0.1
    A simple straight forward approach using sklearn library
"""


def execute(X_train, y_train, X_test):

    # Using Kernel Ridge regression with rbf kernel
    kernel_reg = KernelRidge(kernel='rbf', alpha=1.0)

    kernel_reg.fit(X_train, y_train)

    # Predict values using above fit model
    y_test = kernel_reg.predict(X_test)


    return y_test
