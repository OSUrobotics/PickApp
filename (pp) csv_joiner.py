"""
This code turns a time series data and transforms it into smaller chunks of data that can be used
"""
# @Time : 1/26/2022 11:09 AM
# @Author : Alejandro Velasquez

# System related packages
import os
# File related packages
import csv
import pandas as pd
# Math related packages
import statistics as st
import math
import numpy as np
from numpy import genfromtxt


def labeled_lists(location):
    """
    Sweeps the files within a folder and outputs a list for each label
    :type location: folder with the metadata csvs
    """
    successful_pics = []
    failed_pics = []
    for filename in sorted(os.listdir(location)):

        # First only the metadata files that contain the 'successful' and 'failure' label
        if filename.endswith('_metadata.csv'):

            rows = []
            with open(original_location + filename) as csv_file:
                # Create  a csv object
                csv_reader = csv.reader(csv_file, delimiter=',')
                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)
                # Read the label
                if rows[1][10] == 's':
                    successful_pics.append(filename)
                elif rows[1][10] == 'f':
                    failed_pics.append(filename)

    return successful_pics, failed_pics


# --- Step 1: Read the metadata-files and make a list of the successful picks and failure picks

# Original location where the raw data is located
# original_location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/Apple Proxy Picks/Winter 2022/1_data_valid_for_grasp_and_pick/'
original_location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/Apple Proxy Picks/Winter 2022/2_data_valid_for_grasp/'

success_list, failed_list = labeled_lists(original_location)
print('There are %i experiments with failed apple-picks' % len(failed_list))
print('There are %i experiments with successful apple-picks' % len(success_list))


# --- Step 2: Open the folder respective of each metadata file, that contains the csv files with the data
for pick in success_list:

    # Folder with the csvs that are going to be joined
    location = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess2 (only grasp _ down sampled)/'

    name = str(pick)
    name = name.replace('_metadata.csv', '', 1)

    # First Open the wrench
    topics = ['wrench', 'f1_imu', 'f1_states', 'f2_imu', 'f2_states', 'f3_imu', 'f3_states']

    data_0 = pd.read_csv(location + name + '_grasp_' + topics[0] + '.csv', header=None)
    data_1 = pd.read_csv(location + name + '_grasp_' + topics[1] + '.csv', header=None)
    data_2 = pd.read_csv(location + name + '_grasp_' + topics[2] + '.csv', header=None)
    data_3 = pd.read_csv(location + name + '_grasp_' + topics[3] + '.csv', header=None)
    data_4 = pd.read_csv(location + name + '_grasp_' + topics[4] + '.csv', header=None)
    data_5 = pd.read_csv(location + name + '_grasp_' + topics[5] + '.csv', header=None)
    data_6 = pd.read_csv(location + name + '_grasp_' + topics[6] + '.csv', header=None)

    df = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6], axis=1)

    headers = ['wrench_time', 'force_x', 'force_y', 'force_z', 'net_force', 'torque_x', 'torque_y', 'torque_z',
               'net_torque',
               'f1_imu_time', 'f1_acc_x', 'f1_acc_y', 'f1_acc_z', 'f1_acc_net', 'f1_gyro_x', 'f1_gyro_y', 'f1_gyro_z',
               'f1_states_time', 'f1_position', 'f1_speed', 'f1_effort',
               'f2_imu_time', 'f2_acc_x', 'f2_acc_y', 'f2_acc_z', 'f2_acc_net', 'f2_gyro_x', 'f2_gyro_y', 'f2_gyro_z',
               'f2_states_time', 'f2_position', 'f2_speed', 'f2_effort',
               'f3_imu_time', 'f3_acc_x', 'f3_acc_y', 'f3_acc_z', 'f3_acc_net', 'f3_gyro_x', 'f3_gyro_y', 'f3_gyro_z',
               'f3_states_time', 'f3_position', 'f3_speed', 'f3_effort'
               ]
    df.columns = headers

    location = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess3 (only grasp_downsampled_ joined)/successful/'

    new_file_name = location + name + '_grasp' + '.csv'
    df.to_csv(new_file_name)


# --- Step 3: Do the same with the failed ones
for pick in failed_list:

    # Folder with the csvs that are going to be joined
    location = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess2 (only grasp _ down sampled)/'

    name = str(pick)
    name = name.replace('_metadata.csv', '', 1)

    # First Open the wrench
    topics = ['wrench', 'f1_imu', 'f1_states', 'f2_imu', 'f2_states', 'f3_imu', 'f3_states']

    data_0 = pd.read_csv(location + name + '_grasp_' + topics[0] + '.csv', header=None)
    data_1 = pd.read_csv(location + name + '_grasp_' + topics[1] + '.csv', header=None)
    data_2 = pd.read_csv(location + name + '_grasp_' + topics[2] + '.csv', header=None)
    data_3 = pd.read_csv(location + name + '_grasp_' + topics[3] + '.csv', header=None)
    data_4 = pd.read_csv(location + name + '_grasp_' + topics[4] + '.csv', header=None)
    data_5 = pd.read_csv(location + name + '_grasp_' + topics[5] + '.csv', header=None)
    data_6 = pd.read_csv(location + name + '_grasp_' + topics[6] + '.csv', header=None)

    df = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6], axis=1)

    headers = ['wrench_time', 'force_x', 'force_y', 'force_z', 'net_force', 'torque_x', 'torque_y', 'torque_z',
               'net_torque',
               'f1_imu_time', 'f1_acc_x', 'f1_acc_y', 'f1_acc_z', 'f1_acc_net', 'f1_gyro_x', 'f1_gyro_y', 'f1_gyro_z',
               'f1_states_time', 'f1_position', 'f1_speed', 'f1_effort',
               'f2_imu_time', 'f2_acc_x', 'f2_acc_y', 'f2_acc_z', 'f2_acc_net', 'f2_gyro_x', 'f2_gyro_y', 'f2_gyro_z',
               'f2_states_time', 'f2_position', 'f2_speed', 'f2_effort',
               'f3_imu_time', 'f3_acc_x', 'f3_acc_y', 'f3_acc_z', 'f3_acc_net', 'f3_gyro_x', 'f3_gyro_y', 'f3_gyro_z',
               'f3_states_time', 'f3_position', 'f3_speed', 'f3_effort'
               ]
    df.columns = headers

    location = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess3 (only grasp_downsampled_ joined)/failed/'

    new_file_name = location + name + '_grasp' + '.csv'
    df.to_csv(new_file_name)