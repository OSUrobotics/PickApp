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
        # if filename.endswith('_metadata.csv'):
        if filename.endswith('.csv'):

            rows = []
            with open(location + filename) as csv_file:
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

# Original location where the metadata data is located
main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
# dataset = '1_proxy_rob537_x1/'
# dataset = '2_proxy_rob537_x5/'
dataset = '4_proxy_winter22_x5/'

metadata_loc = main + dataset + 'metadata/'

success_list, failed_list = labeled_lists(metadata_loc)

print('There are %i experiments with failed apple-picks' % len(failed_list))
print('There are %i experiments with successful apple-picks' % len(success_list))


# --- Step 2: Open the folder respective of each metadata file, that contains the csv files with the data
for pick in success_list:

    # Folder with the csvs that are going to be joined
    case = 'GRASP/'
    # case = 'PICK/'
    location = main + dataset + case + 'pp2_downsampled/'

    name = str(pick)

    # For Real-Apple metadata
    # start = name.index('r')
    # end = name.index('k')
    # end_2 = name.index('n')
    # name = name[start:end+1] + '_' + name[end + 1:end_2 - 1]

    # For the rob-537 picks
    # end = name.index('k')
    # end_2 = name.index('n')
    # name = name[:end + 1] + '_' + name[end + 1:end_2 - 1]

    # For the winter-22 picks
    end = name.index('k')
    end_2 = name.index('m')
    name = name[:end + 1] + '' + name[end + 1:end_2 - 1]


    print(name)

    if case == 'GRASP/':
        stage = '_grasp_'
    elif case == 'PICK/':
        stage = '_pick_'

    # For each of the augmentd byte
    for j in range(5):

        # First Open the wrench
        # topics = ['wrench', 'f1_imu', 'f1_states', 'f2_imu', 'f2_states', 'f3_imu', 'f3_states']
        topics = ['wrench_', 'f1_imu_', 'f1_states_', 'f2_imu_', 'f2_states_', 'f3_imu_', 'f3_states_']

        data_0 = pd.read_csv(location + name + stage + topics[0] + str(j) + '.csv', header=None)
        data_1 = pd.read_csv(location + name + stage + topics[1] + str(j) + '.csv', header=None)
        data_2 = pd.read_csv(location + name + stage + topics[2] + str(j) + '.csv', header=None)
        data_3 = pd.read_csv(location + name + stage + topics[3] + str(j) + '.csv', header=None)
        data_4 = pd.read_csv(location + name + stage + topics[4] + str(j) + '.csv', header=None)
        data_5 = pd.read_csv(location + name + stage + topics[5] + str(j) + '.csv', header=None)
        data_6 = pd.read_csv(location + name + stage + topics[6] + str(j) + '.csv', header=None)

        # data_0 = pd.read_csv(location + name + stage + topics[0] + '.csv', header=None)
        # data_1 = pd.read_csv(location + name + stage + topics[1] + '.csv', header=None)
        # data_2 = pd.read_csv(location + name + stage + topics[2] + '.csv', header=None)
        # data_3 = pd.read_csv(location + name + stage + topics[3] + '.csv', header=None)
        # data_4 = pd.read_csv(location + name + stage + topics[4] + '.csv', header=None)
        # data_5 = pd.read_csv(location + name + stage + topics[5] + '.csv', header=None)
        # data_6 = pd.read_csv(location + name + stage + topics[6] + '.csv', header=None)


        df = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6], axis=1)


        # headers = ['time', 'force_x', 'force_y', 'force_z', 'net_force', 'torque_x', 'torque_y', 'torque_z', 'net_torque',
        #            'time', 'f1_acc_x', 'f1_acc_y', 'f1_acc_z', 'net_acc', 'f1_gyro_x', 'f1_gyro_y', 'f1_gyro_z',
        #            'time', 'f1_position', 'f1_speed', 'f1_effort',
        #            'time', 'f2_acc_x', 'f2_acc_y', 'f2_acc_z', 'net_acc', 'f2_gyro_x', 'f2_gyro_y', 'f2_gyro_z',
        #            'time', 'f2_position', 'f2_speed', 'f2_effort',
        #            'time', 'f3_acc_x', 'f3_acc_y', 'f3_acc_z', 'net_acc', 'f3_gyro_x', 'f3_gyro_y', 'f3_gyro_z',
        #            'time', 'f3_position', 'f3_speed', 'f3_effort'
        #            ]
        headers = ['force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z',
                   'f1_acc_x', 'f1_acc_y', 'f1_acc_z', 'f1_gyro_x', 'f1_gyro_y', 'f1_gyro_z',
                   'f1_position', 'f1_speed', 'f1_effort',
                   'f2_acc_x', 'f2_acc_y', 'f2_acc_z', 'f2_gyro_x', 'f2_gyro_y', 'f2_gyro_z',
                   'f2_position', 'f2_speed', 'f2_effort',
                   'f3_acc_x', 'f3_acc_y', 'f3_acc_z', 'f3_gyro_x', 'f3_gyro_y', 'f3_gyro_z',
                   'f3_position', 'f3_speed', 'f3_effort'
                   ]
        df.columns = headers

        # Target Location
        target_location = main + dataset + case + 'pp3_joined/'
        new_file_name = target_location + name + '_' + str(stage) + str(j) + '.csv'
        df.to_csv(new_file_name)

