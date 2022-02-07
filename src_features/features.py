import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures

from tsfresh import extract_relevant_features
import numpy as np
import pandas as pd
from pandas import Int64Index as NumericIndex
import csv
import tqdm

from numpy import genfromtxt


def get_features(location):
    # Read the example from ts-fresh
    # timeseries = pd.read_csv(location + 'example_x.csv')
    # y = pd.read_csv(location + 'example_y.csv', index_col=0, squeeze=True)

    # Read data from apple-picks
    timeseries = pd.read_csv(location + 'joined_timeseries.csv')
    # y = pd.read_csv(location + 'y_smaller.csv', index_col=False, header=None, squeeze=True)
    y = pd.read_csv(location + 'joined_y.csv', index_col=0, header=0, squeeze=True)

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

    extracted = extract_features(timeseries, column_id="id", column_sort="time")
    extracted.to_csv(location + 'features.csv')

    impute(extracted)
    X_filtered = select_features(extracted, y)
    print('--- The extracted features are:', extracted)
    print('--- The extracted filtered features are: ', X_filtered)

    X_filtered.to_csv(location + 'features.csv')

    return X_filtered


def best_features(features):
    """
    From the huge list of features extracted by ts-fresh, this function
    leaves only the first feature of each variable
    :param features: features extracted with ts-fresh
    :return: Columns that contain the first feature of each variable
    """
    variables_checked = []
    variable_and_features = []
    cols = []
    col = 0

    for i in list(features.columns.values):

        variable = str(i)
        end = variable.find('__')  # Every feature starts with double __
        variable = variable[:end]
        print(variable)

        if not variable in variables_checked:
            cols.append(col)
            variables_checked.append(variable)
            variable_and_features.append(i)

        col += 1

    print(variable_and_features)
    print(variables_checked)
    print(cols)

    return cols


def create_list(location, features, cols):
    hope = pd.DataFrame()
    s = 0
    for col in cols:
        feature = features.iloc[:, col]
        hope[s] = feature
        # print(feature)
        s += 1

    print(hope)
    hope.to_csv(location + 'best_features.csv', index=False)


def tsfresh_example():
    """
    Runs the example from the website https://tsfresh.readthedocs.io/en/latest/text/quick_start.html
    """

    # ---- Download Data ---- #
    download_robot_execution_failures()
    timeseries, y = load_robot_execution_failures()


    # location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_4_for_tsfresh/'
    # timeseries.to_csv(location + 'example_x.csv')
    # y.to_csv(location + 'example_y.csv')

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




if __name__ == "__main__":
    # Load data
    location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_4_for_tsfresh/'

    # Obtain the filtered features
    # features = get_features(location)

    # Leave only the most important features
    # features = pd.read_csv(location + 'features.csv')
    # print(features)

    # columns = best_features(features)
    # create_list(location, features, columns)

    # Run the ts-fresh example as a reference
    tsfresh_example()