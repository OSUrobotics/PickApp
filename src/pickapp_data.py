# @Time : 4/7/2022 11:15 AM
# @Author : Alejandro Velasquez
"""
This script performs the data post-processing, before feeding it into any machine learning algorithm
"""

import os
import numpy as np
from numpy import genfromtxt
import math
import statistics as st
import matplotlib.pyplot as plt
from statistics import mean, stdev
import pandas as pd
from tqdm import tqdm
import csv
import shutil
import random

# trial 2

def check_size(source):

    lowest = 10000
    highest = 0
    sizes = []
    for filename in tqdm(os.listdir(source)):

        data = pd.read_csv(source + filename)
        n_samples = data.shape[0]
        sizes.append(n_samples)

        if n_samples < lowest:
            lowest = n_samples

        if n_samples > highest:
            highest = n_samples

    title = "Lowest= " + str(lowest) + " / Highest= " + str(highest) + " / Mean=" + str(round(mean(sizes),2)) + " / SD= " + str(round(stdev(sizes),2))
    plt.title(title)
    plt.boxplot(sizes)
    plt.show()

    return lowest, highest


def down_sample(period, source, target):
    """
    Downsamples all the csv files located in source folder, and saves the new csv in target folder
    :param period: period [ms] at which you want to sample the time series
    :param source: subfolder with original data
    :param target: subfolder to save the downsampled data
    :return:
    """

    for filename in os.listdir(source):
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
    """
    Joins csv from different topics but from the same experiment, into a single csv.
    Thus, data is easier to handle, and less prone to make mistakes.
    It does some cropping of the initial or last points, in order to have all the topics have the same size
    :param name: Name of the dataset / experiment
    :param case: Whether Grasp or Pick stage
    :param source:
    :param target:
    :return:
    """

    if case == 'GRASP/':
        stage = 'grasp'
    elif case == 'PICK/':
        stage = 'pick'

    # --- Step 1: Open all the topics from the same experiment that need to be joined ---
    location = source
    topics = ['_wrench', '_f1_imu', '_f1_states', '_f2_imu', '_f2_states', '_f3_imu', '_f3_states']

    data_0 = pd.read_csv(location + name + stage + topics[0] + '.csv', header=None, index_col=False)
    data_1 = pd.read_csv(location + name + stage + topics[1] + '.csv', header=None, index_col=False)
    data_2 = pd.read_csv(location + name + stage + topics[2] + '.csv', header=None, index_col=False)
    data_3 = pd.read_csv(location + name + stage + topics[3] + '.csv', header=None, index_col=False)
    data_4 = pd.read_csv(location + name + stage + topics[4] + '.csv', header=None, index_col=False)
    data_5 = pd.read_csv(location + name + stage + topics[5] + '.csv', header=None, index_col=False)
    data_6 = pd.read_csv(location + name + stage + topics[6] + '.csv', header=None, index_col=False)

    dataframes = [data_0, data_1, data_2, data_3, data_4, data_5, data_6]

    # --- Step 2: Crop initial or last points in order to make all topics have the same length
    # Get the channel with the less sampled points
    smallest = 10000
    for channel in dataframes:
        if channel.shape[0] < smallest:
            smallest = channel.shape[0]
            benchmark = channel
    # print("\nSmallest:", smallest)

    benchmark_first = float(benchmark.iloc[1, 0])  # First Reading
    benchmark_last = float(benchmark.iloc[-1, 0])  # Last Reading
    # print("First and last", benchmark_first, benchmark_last)

    count = 0

    for channel in dataframes:
        if channel.shape[0] > smallest:
            difference = channel.shape[0] - smallest
            # print("The difference is", difference)

            if difference > 5:
                pass
                # If difference of sampled points is bigger than a threshold value, then print this warning to manually
                # check it.
                # print("//////////////////////////// WARNING ///////////////////////////")

            # Decide which points to crop: the initial or last ones
            initial_time_offset = abs(float(channel.iloc[1, 0]) - benchmark_first)
            last_time_offset = abs(float(channel.iloc[-1, 0]) - benchmark_last)

            if initial_time_offset > last_time_offset:
                # print("Remove initial")
                for i in range(difference):
                    new_df = channel.drop([1]).reset_index(drop=True)
                    channel = new_df
            else:
                # print("Remove last")
                new_df = channel.iloc[:-difference, :]

            dataframes[count] = new_df

        count = count + 1

    # --- Step 3: Join dataframes from each topic into a single dataframe and save
    df = pd.concat([dataframes[0].iloc[:, 1:], dataframes[1].iloc[:, 1:], dataframes[2].iloc[:, 1:],
                    dataframes[3].iloc[:, 1:], dataframes[4].iloc[:, 1:], dataframes[5].iloc[:, 1:],
                    dataframes[6].iloc[:, 1:]], axis=1)
    new_file_name = target + name + '_' + str(stage) + '.csv'
    df.to_csv(new_file_name, index=False, header=False)


def crop_csv(size, source, target):

    for filename in os.listdir(source):

        data = pd.read_csv(source + filename)
        n_samples = data.shape[0]
        difference = n_samples - size
        start = int(difference/2)
        end = start + size
        cropped_data = data.iloc[start:end, :]
        cropped_data.to_csv(target + filename, index=False)




def noise_injection(data, percentage):
    """
    Data augmentation technique that simply adds noise to the signal as a random Gaussian noise
    :param percentage: Percentage of the range (Max - Min) of the signal that would be considered in the noise function
    :type data: Dataframe
    :return: New datafram with noise
    """

    channels = data.shape[1]

    df = pd.DataFrame()

    for i in range(channels):
        channel = data.iloc[:, i]

        # Step 1 - Read min max for each column
        channel_range = abs(min(channel) - max(channel))

        # Step 2 - Define a % of noise according to that range
        noise = np.random.normal(0, channel_range * percentage/100, channel.shape)
        new_signal = channel + noise

        df[i] = new_signal

    # Copy original dataframe header
    df.columns = data.columns

    return df


def data_into_labeled_folder(dataset, metadata_location, data_source_folder, target_folder):
    """
    Distribute the csv files in labeled folders
    :param dataset:
    :param metadata_location: Folder with metadata files, which have the labels of the experiments
    :param data_source_folder:
    :param target_folder:
    :return:
    """

    for metadata in (os.listdir(metadata_location)):

        # --- Step 1: Get the basic name
        name = str(metadata)

        if dataset == '1_proxy_rob537_x1/':
            start = name.index('app')
            end = name.index('k')
            end_2 = name.index('m')
            name = name[start:end + 1] + '_' + name[end + 1:end_2 - 1] + '_'

        elif dataset == '3_proxy_winter22_x1/':
            start = name.index('app')
            end = name.index('m')
            name = name[start:end]

        elif dataset == '5_real_fall21_x1/':
            start = name.index('r')
            end = name.index('k')
            end_2 = name.index('m')
            name = name[start:end+1] + '_' + name[end + 1:end_2 - 1] + '_'

        # --- Step 2: Read label / result from metadata
        rows = []
        with open(metadata_location + metadata) as csv_file:
            # Create a csv object
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Extract each data row one by one
            for row in csv_reader:
                rows.append(row)
            # Read the label
            if rows[1][10] == 's':
                sub_folder = 'success/'
            else:
                sub_folder = 'failed/'

        # --- Step 3: S

        for filename in os.listdir(data_source_folder):

            data_name = str(filename)
            end = data_name.index("__")
            data_name = data_name[:end+1]

            if name == data_name:
                # print("Meta and data names:", name, data_name)
                # print("\n\n\n\n\n\n\nMatch!")
                source = data_source_folder + filename
                target = target_folder + sub_folder +  filename
                shutil.copy(source, target)


def create_sets(main, dataset, training_size):
    """
    Distributes the data in a Training and Testing set, by keeping the same label ratios
    :param main:
    :param dataset:
    :param training_size: Size of training set from 0 to 1, the remaining goes to the test set
    :return:
    """

    # Make sure that the augmented data is not divided into training and testing set, otherwise the testing wouldn't
    # take place with unseen data.


    # stages = ['GRASP/', 'PICK/']
    labels = ['failed/', 'success/']
    augmented_folders = ['augmented x5/', 'augmented x10/', 'augmented x20/']

    for augmented_folder in augmented_folders:

        for label in labels:

            grasp_source_location = main + dataset + 'GRASP/' + 'new_pp5_labeled/' + augmented_folder + label
            pick_source_location = main + dataset + 'PICK/' + 'new_pp5_labeled/' + augmented_folder + label

            previous_name = ''
            for filename in os.listdir(grasp_source_location):

                # print(filename)
                name = str(filename)
                end = name.index('grasp')
                name = name[:end]
                # print(name)

                if name != previous_name:
                    # Check name with previous, if different, flip coin
                    coin = random.random()
                    print(coin)

                    if coin < training_size:
                        grasp_target_location = main + dataset + 'GRASP/' + 'new_pp6_sets/' + augmented_folder + 'training set/' + label
                        pick_target_location = main + dataset + 'PICK/' + 'new_pp6_sets/' + augmented_folder + 'training set/' + label
                    else:
                        grasp_target_location = main + dataset + 'GRASP/' + 'new_pp6_sets/' + augmented_folder + 'testing set/' + label
                        pick_target_location = main + dataset + 'PICK/' + 'new_pp6_sets/' + augmented_folder + 'testing set/' + label

                    previous_name = name
                    # print(previous_name)
                else:
                    pass

                # --- Move data from the Grasp ---
                original = grasp_source_location + filename
                target = grasp_target_location + filename
                shutil.copy(original, target)

                # And from the Pick
                filename = filename.replace('grasp', 'pick')
                original = pick_source_location + filename
                target = pick_target_location + filename
                shutil.copy(original, target)


def main():

    # Step 1 - Read Data saved as csvs from bagfiles

    # Step 2 - Split the data into Grasp and Pick
    # (pp) grasp_and_pick_split.py

    # Step 3 - Select the columns to pick
    # (pp) real_pick_delCol.py

    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'

    # dataset = '1_proxy_rob537_x1/'
    dataset = '3_proxy_winter22_x1/'
    # dataset = '5_real_fall21_x1/'

    stages = ['GRASP/', 'PICK/']

    print("\nStep 1: Downsampling...")
    # for stage in tqdm(stages):
    #     location = main + dataset + stage
    #     location_1 = location + 'pp1_split/'
    #     location_2 = location + 'new_pp2_downsampled/'
    #
    #     # --- Step 4: Down sample Data ---
    #     period = 15  # Sampling period in [ms]
    #
    #     down_sample(period, location_1, location_2)
    #
    #     # --- Step 5: Check sizes ---
    #     # check_size(location_2)

    # --- Step 6: Join Data ---
    # Here we want to end up with a list the size of the medatadafiles
    # Thus makes sense to get the names from the metadata folder
    # (pp) csv_joiner.py
    metadata_loc = main + dataset + 'metadata/'

    print("\nStep 2: Joining topics into a single csv...")
    # for filename in tqdm(sorted(os.listdir(metadata_loc))):
    #
    #     # Get the basic name
    #     name = str(filename)
    #
    #     if dataset == '1_proxy_rob537_x1/':
    #         start = name.index('app')
    #         end = name.index('k')
    #         end_2 = name.index('m')
    #         name = name[start:end + 1] + '_' + name[end + 1:end_2 - 1] + '_'
    #
    #     elif dataset == '3_proxy_winter22_x1/':
    #         start = name.index('app')
    #         end = name.index('m')
    #         name = name[start:end]
    #
    #     elif dataset == '5_real_fall21_x1/':
    #         start = name.index('r')
    #         end = name.index('k')
    #         end_2 = name.index('m')
    #         name = name[start:end+1] + '_' + name[end + 1:end_2 - 1] + '_'
    #
    #     # print("\nFiles being checked:")
    #     # print(filename)
    #     # print(name)
    #
    #     for stage in stages:
    #         # print(stage)
    #         location = main + dataset + stage
    #         location_2 = location + 'new_pp2_downsampled/'
    #         location_3 = location + 'new_pp3_joined/'
    #         join_csv(name, stage, location_2, location_3)

    # --- Step 7: Augment Data ---

    print("\n Step 3: Augmenting data...")
    # for stage in tqdm(stages):
    #     location = main + dataset + stage
    #     location_3 = location + 'new_pp3_joined/'
    #     location_4 = location + 'new_pp4_augmented/augmented x20/'
    #
    #     for filename in os.listdir(location_3):
    #         # print(filename)
    #
    #         data = pd.read_csv(location_3 + filename)
    #         augmentations = 20
    #         end = filename.index('.')
    #         for i in range(augmentations):
    #             augmented_data = noise_injection(data, augmentations)
    #             new_name = filename[:end] + "_aug_" + str(i) + ".csv"
    #             augmented_data.to_csv(location_4 + new_name, index=False)

    # --- Step 8: Save csvs in subfolders labeled ---

    print("\nStep 4: Saving data in labeled folders...")
    # for stage in tqdm(stages):
    #     location = main + dataset + stage
    #
    #     if dataset in ['1_proxy_rob537_x1/', '3_proxy_winter22_x1/']:
    #         location_4 = location + 'new_pp4_augmented/augmented x20/'
    #     elif dataset == '5_real_fall21_x1/':
    #         location_4 = location + 'new_pp3_joined/'
    #
    #     location_5 = location + 'new_pp5_labeled/augmented x20/'
    #     metadata_loc = main + dataset + 'metadata/'
    #
    #     data_into_labeled_folder(dataset, metadata_loc, location_4, location_5)


    # --- Step 9: Sparse data in the training and testing set

    print("\nStep 5: Sparsing data in training and testing sets...")
    training_size = 0.7
    create_sets(main, dataset, training_size)


if __name__ == '__main__':
    main()