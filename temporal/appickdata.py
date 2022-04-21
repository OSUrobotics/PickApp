# @Time : 4/7/2022 11:15 AM
# @Author : Alejandro Velasquez

from sklearn.utils import resample

import os
import numpy as np
from numpy import genfromtxt
import math
import statistics as st
import matplotlib.pyplot as plt
from statistics import mean
import pandas as pd
from tqdm import tqdm
import csv


def check_size(source):

    lowest = 10000
    highest = 0
    for filename in tqdm(os.listdir(source)):

        data = pd.read_csv(source + filename)
        n_samples = data.shape[0]

        if n_samples < lowest:
            lowest = n_samples

        if n_samples > highest:
            highest = n_samples

    return lowest, highest


def down_sample(period, source, target):
    """
    Downsamples all the csv files located in source folder, and saves the new csv in target folder
    :param period: period [ms] at which you want to sample the time series
    :param source:
    :param target:
    :return:
    """

    for filename in tqdm(os.listdir(source)):
        # print(filename)

        # --- Step 0: Read csv data into a a Pandas Dataframe ---
        # Do not include the first column that has the time, so we don't overfit the next processes
        # data = genfromtxt((source + filename), delimiter=',', skip_header=True)

        data = pd.read_csv(source + filename)
        n_samples = data.shape[0]       # rows
        n_channels = data.shape[1]      # columns

        max_time = data.iloc[-1, 0]

        # Create New Dataframe
        downsampled_data = pd.DataFrame()
        headers = pd.read_csv(source + filename, index_col=0, nrows=0).columns.tolist()

        # print(headers)

        for i in range(n_channels):
            new_value = []
            if i == 0:
                # --- Time Channel
                new_time = []

                time = data.iloc[0, 0]
                while time < max_time:
                    new_time.append(time)
                    time = time + period/1000
                    # print(time)
                header = "Time"
                downsampled_data[header] = new_time

            else:
                # --- The rest of the channels
                new_value = []
                index = 0
                for x in new_time:
                    for k in data.iloc[index:, 0]:
                        if k > x:
                            break
                        else:
                            index += 1

                    # Interpolation
                    x1 = data.iloc[index-1, 0]
                    x2 = data.iloc[index, 0]
                    y1 = data.iloc[index-1, i]
                    y2 = data.iloc[index, i]
                    value = (y1 - y2)*(x2 - x)/(x2 - x1) + y2
                    new_value.append(value)

                    header = headers[i-1]

                downsampled_data[header] = new_value

                # --- Compare PLots ---
                # plt.plot(data.iloc[:, 0], data.iloc[:, i])
                # plt.plot(new_time, new_value)
                # plt.show()

        # print(downsampled_data)
        downsampled_data.to_csv(target + filename, index=False)


def join_csv(name, case, source, target):

    if case == 'GRASP/':
        stage = 'grasp'
    elif case == 'PICK/':
        stage = 'pick'

    location = source

    # Open all the topics to join
    topics = ['_wrench', '_f1_imu', '_f1_states', '_f2_imu', '_f2_states', '_f3_imu', '_f3_states']

    data_0 = pd.read_csv(location + name + stage + topics[0] + '.csv', header=None, index_col=False).iloc[:, 1:]
    data_1 = pd.read_csv(location + name + stage + topics[1] + '.csv', header=None, index_col=False).iloc[:, 1:]
    data_2 = pd.read_csv(location + name + stage + topics[2] + '.csv', header=None, index_col=False).iloc[:, 1:]
    data_3 = pd.read_csv(location + name + stage + topics[3] + '.csv', header=None, index_col=False).iloc[:, 1:]
    data_4 = pd.read_csv(location + name + stage + topics[4] + '.csv', header=None, index_col=False).iloc[:, 1:]
    data_5 = pd.read_csv(location + name + stage + topics[5] + '.csv', header=None, index_col=False).iloc[:, 1:]
    data_6 = pd.read_csv(location + name + stage + topics[6] + '.csv', header=None, index_col=False).iloc[:, 1:]

    # Join them all into a single DataFrame
    df = pd.concat([data_0, data_1, data_2, data_3, data_4, data_5, data_6], axis=1)

    # Target Location
    new_file_name = target + name + '_' + str(stage) + '.csv'
    df.to_csv(new_file_name, index=False, header=False)


def crop_csv(size, source, target):

    for filename in tqdm(os.listdir(source)):

        data = pd.read_csv(source + filename)
        n_samples = data.shape[0]
        difference = n_samples - size
        start = int(difference/2)
        end = start + size
        cropped_data = data.iloc[start:end, :]
        cropped_data.to_csv(target + filename, index=False)


def main():
    # Step 1 - Read Data saved as csvs from bagfiles
    # Step 2 - Split the data into Grasp and Pick
    # (pp) grasp_and_pick_split.py

    # Step 3 - Select the columns to pick
    # (pp) real_pick_delCol.py

    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    dataset = '3_proxy_winter22_x1/'
    # dataset = '5_real_fall21_x1/'
    # dataset = '1_proxy_rob537_x1/'
    stages = ['GRASP/', 'PICK/']

    # (pp) downsampler.py
    for stage in tqdm(stages):
        location = main + dataset + stage
        location_1 = location + 'pp1_split/'
        location_2 = location + 'new_pp2_downsampled/'
        location_3 = location + 'new_pp3_cropped/'
        location_4 = location + 'new_pp4_joined/'
        location_5 = location + 'new_pp5_labeled/'

        # --- Step 4: Down sample Data ---
        period = 15  # Sampling period in [ms]
        # down_sample(period, location_1, location_2)

        # --- Step 5: Crop Data ---
        # l, h = check_size(location_2)
        # print(dataset, stage, l, h)
        if stage == 'GRASP/':
            size = 106
        elif stage == 'PICK/':
            size = 115
        # crop_csv(size, location_2, location_3)

    # --- Step 6: Join Data ---
    # Here we want to end up with a list the size of the medatadafiles
    # Thus makes sense to get the names from the metadata folder
    # (pp) csv_joiner.py
    metadata_loc = main + dataset + 'metadata/'

    for filename in tqdm(sorted(os.listdir(metadata_loc))):

        # Get the basic name
        name = str(filename)
        start = name.index('app')
        end = name.index('m')
        name = name[start:end]
        # print(name)

        for stage in stages:
            location = main + dataset + stage
            location_3 = location + 'new_pp3_cropped/'
            location_4 = location + 'new_pp4_joined/'
            join_csv(name, stage, location_3, location_4)


    # --- Step 7: Augment Data ---


    # Step 6 - Do Data Augmentation by adding Noise
    # TODO

    # Step 7 - Save csvs in subfolders labeled
    # (pp) data_into_labeled_folder.py


if __name__ == '__main__':
    main()