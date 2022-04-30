# tsfresh example

import matplotlib.pyplot as plt
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_relevant_features
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, \
    load_robot_execution_failures
download_robot_execution_failures()
timeseries, y = load_robot_execution_failures()

print(timeseries.head())


timeseries[timeseries['id'] == 20].plot(subplots=True, sharex=True, figsize=(10, 10))
plt.show()



extracted_features = extract_features(timeseries, column_id="id", column_sort="time")

impute(extracted_features)
features_filtered = select_features(extracted_features, y)

features_filtered_direct = extract_relevant_features(timeseries, y,
                                                     column_id='id', column_sort='time')


print(extracted_features)