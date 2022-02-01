# @Time : 1/30/2022 6:51 PM
# @Author : Alejandro Velasquez

import pandas as pd

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":

    # ---- Download Data ---- #
    download_robot_execution_failures()
    timeseries, y = load_robot_execution_failures()

    location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_4_for_tsfresh/'
    timeseries.to_csv(location + 'example_x.csv')
    y.to_csv(location + 'example_y.csv')

    # Time Series Data
    print('--- Time series data ---')
    print('Type: ', type(timeseries))
    print('Length: ', len(timeseries))
    print('Data: \n', timeseries)


    # Results Data
    print('--- Results data ---')
    print('Type: ', type(y))
    print('Length: ', len(y))
    print('Data: \n', y)

    # ---- Extract Features ---- #
    X = extract_features(timeseries, column_id="id", column_sort="time")
    impute(X)
    X_filtered = select_features(X, y)
    print(X_filtered)
    # X_filtered.to_csv('look_at_me.csv')

    # ---- Machine Learning ---- #
    # # Train and evaluate the classifier
    # X_full_train, X_full_test, y_train, y_test = train_test_split(X, y, test_size=.4)
    # X_filtered_train, X_filtered_test = X_full_train[X_filtered.columns], X_full_test[X_filtered.columns]
    #
    # classifier_full = DecisionTreeClassifier()
    # classifier_full.fit(X_full_train, y_train)
    # print(classification_report(y_test, classifier_full.predict(X_full_test)))
    #
    # classifier_filtered = DecisionTreeClassifier()
    # classifier_filtered.fit(X_filtered_train, y_train)
    # print(classification_report(y_test, classifier_filtered.predict(X_filtered_test)))
