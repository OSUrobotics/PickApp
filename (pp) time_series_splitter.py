"""
This code turns a time series data and transforms it into smaller chunks of data that can be used
"""

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
original_location = '/media/avl/StudyData/Apple Pick Data/Apple Proxy Picks/3 - Winter 2022/1_data_valid_for_grasp_and_pick/'
# Target location to store the post-processed data
target_location = '/home/avl/PycharmProjects/AppleProxy_PostProcessed_Data/'
success_list, failed_list = labeled_lists(original_location)

print('There are %i experiments with failed apple-picks' % len(failed_list))
print('There are %i experiments with successful apple-picks' % len(success_list))


# --- Step 2: Open the folder respective of each metadata file, that contains the csv files with the data
for pick in success_list:

    folder = str(pick)
    folder = folder.replace('_metadata.csv', '', 1)

    for filename in os.listdir(original_location + folder):
        data = genfromtxt((original_location + folder + '/' + filename), delimiter=',', skip_header=True)
        size = len(data) - 1
        print('\nThe initial number of point of the time series is: ', size)

        print(filename)



location = 'csvs/successful_picks/'
# location = 'csvs/failed_picks/'

lengths = []

# Divisions to split the time series
divisions = 18

for filename in os.listdir(location):
    print(filename)

    # ---- Step 0: Read csv data into an array -----
    data = genfromtxt((location + filename), delimiter=',', skip_header=True)
    # print(data)

    size = len(data) - 1
    print('\nThe initial number of point of the time series is: ', size)
    print('The number of divisions is: ', divisions)

    n_bytes = 5
    points_per_division = math.floor(size / divisions)
    print('The number of points per division is: ', points_per_division)

    window = math.floor(size / (divisions * n_bytes))
    print('The window of data_points is: ', window)

    # --------------- Step 2: Slice data ----------------------------
    for i in range(n_bytes):
        byte = []
        for j in range(divisions):
            start = j * points_per_division + i * window
            end = start + window
            slice = data[start:end]

            # swipe all the columns
            values = []
            for p in range(int(slice.size / window)):
                value = st.mean(slice[:, p])
                values.append(value)

            # print(slice)
            byte.append(values)

        # Create the new data file
        name = 'csvs/new_data (x5)/' + filename[:-4] + '_' + str(i) + '.csv'

        with open(name, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerows(byte)

        # print('The %i-th byte is: ' % i)
        # print(byte)
        # print('\n')
