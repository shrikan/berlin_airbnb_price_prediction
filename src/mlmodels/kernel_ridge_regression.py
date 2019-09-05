import pandas as pd
from sklearn.kernel_ridge import KernelRidge


def execute(X_train, y_train, X_test):

    # Using Kernel Ridge regression with rbf kernel
    kernel_reg = KernelRidge(kernel='rbf', alpha=1.0)

    kernel_reg.fit(X_train, y_train)

    # Predict values using above fit model
    y_test = kernel_reg.predict(X_test)


    return y_test
