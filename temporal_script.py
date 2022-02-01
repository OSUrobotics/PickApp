import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
import numpy as np
import pandas as pd
from pandas import Int64Index as NumericIndex
import csv
import tqdm

from numpy import genfromtxt


if __name__ == "__main__":

    # Load data
    location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_4_for_tsfresh/'

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

    # features_filtered_direct = extract_relevant_features(timeseries, y, column_id='id', column_sort='time')
    # print(features_filtered_direct)