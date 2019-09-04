import pandas as pd
from sklearn.kernel_ridge import KernelRidge


def get_missing_values(df):
    reduced_df = df[['bedrooms', 'price', 'cleaning_fee', 'distance', 'extra_people', 'area']]
    df.drop(['bedrooms', 'price', 'cleaning_fee', 'distance', 'extra_people', 'area'], axis=1, inplace=True)

    training_data = reduced_df[reduced_df['area'].notnull()]
    test_data = reduced_df[reduced_df['area'].isnull()]

    # Training req
    X_train = training_data.drop('area', axis=1)
    y_train = training_data['area']

    # Testing req
    X_test = test_data.drop('area', axis=1)

    # Using Kernel Ridge regression with rbf kernel
    kernel_reg = KernelRidge(kernel='rbf', alpha=1.0)

    kernel_reg.fit(X_train, y_train)

    # Predict values using above fit model
    y_test = kernel_reg.predict(X_test)

    predicted_df = pd.DataFrame(X_test)
    predicted_df['area'] = y_test

    updated_df = pd.concat([predicted_df, training_data], axis=0)

    # To handle unwanted predicted area which  might be completely out of range
    updated_df.drop(updated_df[(updated_df['area'] <= 0.0) |
                                   (updated_df['area'] > 250.0)].index, axis=0, inplace=True)

    merged_df = pd.concat([df, updated_df], axis=1)

    return merged_df
