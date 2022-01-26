"""
Script to join different csv files into a single one, so the data is not spread in many files but rahter in only one.
This happens often in ROS, as there is always a csv per topic.

Sources:
https://www.geeksforgeeks.org/how-to-merge-multiple-csv-files-into-a-single-pandas-dataframe/
"""

import tqdm
import os
import csv
from numpy import genfromtxt

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# Step 1 - Import Data
location = '/home/avl/PycharmProjects/AppleProxy/Data/new_data (x5)/failed_picks/'
name = 'real_apple_pick_'

success = [6, 10,
           16,
           30, 31, 38,
           42, 43, 48,
           50, 51, 52, 53,
           60, 61, 63, 64, 67,
           70, 71, 72, 73, 74, 77]

not_useful = [11, 36, 37, 38, 40, 41, 42]

# Do it for Grasp
for i in range(1, 77):
    # Count how many files

    if not(i in success) or (i in not_useful):
        continue

    # For each of the subdata generated
    for j in range(20):
        # location = '/home/avl/PycharmProjects/AppleProxy/Data/new_data (x20)/failed_picks/'
        location = '/home/avl/PycharmProjects/AppleProxy/Data/new_data (x20)/successful_picks/'
        name = 'real_apple_pick_'

        # First Open the wrench
        topics = ['wrench_', 'f1_imu_', 'f1_states_', 'f2_imu_', 'f2_states_', 'f3_imu_', 'f3_states_']

        data_0 = pd.read_csv(location + name + str(i) + '_grasp_' + topics[0] + str(j) + '.csv', header=None)
        data_1 = pd.read_csv(location + name + str(i) + '_grasp_' + topics[1] + str(j) + '.csv', header=None)
        data_2 = pd.read_csv(location + name + str(i) + '_grasp_' + topics[2] + str(j) + '.csv', header=None)
        data_3 = pd.read_csv(location + name + str(i) + '_grasp_' + topics[3] + str(j) + '.csv', header=None)
        data_4 = pd.read_csv(location + name + str(i) + '_grasp_' + topics[4] + str(j) + '.csv', header=None)
        data_5 = pd.read_csv(location + name + str(i) + '_grasp_' + topics[5] + str(j) + '.csv', header=None)
        data_6 = pd.read_csv(location + name + str(i) + '_grasp_' + topics[6] + str(j) + '.csv', header=None)

        # data_1 = location + name + str(i) + '_grasp_' + topics[1] + str(j) + '.csv'
        # data_2 = location + name + str(i) + '_grasp_' + topics[2] + str(j) + '.csv'
        # data_3 = location + name + str(i) + '_grasp_' + topics[3] + str(j) + '.csv'
        # data_4 = location + name + str(i) + '_grasp_' + topics[4] + str(j) + '.csv'
        # data_5 = location + name + str(i) + '_grasp_' + topics[5] + str(j) + '.csv'
        # data_6 = location + name + str(i) + '_grasp_' + topics[6] + str(j) + '.csv'

        # Concatenate into a single pandas dataframe
        # df = pd.concat(map(pd.read_csv, [data_0, data_1, data_2, data_3, data_4, data_5, data_6]), axis=1)
        df = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6], axis=1)
        # And save it
        headers = ['wrench_time', 'force_x', 'force_y', 'force_z', 'net_force', 'torque_x', 'torque_y', 'torque_z', 'net_torque',
                   'f1_imu_time', 'f1_acc_x', 'f1_acc_y', 'f1_acc_z', 'f1_acc_net', 'f1_gyro_x', 'f1_gyro_y', 'f1_gyro_z',
                   'f1_states_time', 'f1_position', 'f1_speed', 'f1_effort',
                   'f2_imu_time', 'f2_acc_x', 'f2_acc_y', 'f2_acc_z', 'f2_acc_net', 'f2_gyro_x', 'f2_gyro_y', 'f2_gyro_z',
                   'f2_states_time', 'f2_position', 'f2_speed', 'f2_effort',
                   'f3_imu_time', 'f3_acc_x', 'f3_acc_y', 'f3_acc_z', 'f3_acc_net', 'f3_gyro_x', 'f3_gyro_y', 'f3_gyro_z',
                   'f3_states_time', 'f3_position', 'f3_speed', 'f3_effort'
                   ]
        df.columns = headers
        # location = '/home/avl/PycharmProjects/AppleProxy/Data/new_data (x20) concatenated/failed_picks/'
        location = '/home/avl/PycharmProjects/AppleProxy/Data/new_data (x20) concatenated/successful_picks/'

        new_file_name = location + name + str(i) + '_grasp_' + str(j) + '.csv'
        df.to_csv(new_file_name)

        print(df)
        # input()








