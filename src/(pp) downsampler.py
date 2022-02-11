# @Time : 1/26/2022 11:09 AM
# @Author : Alejandro Velasquez

import os
import csv
import statistics as st
import math
import numpy as np
from numpy import genfromtxt
import pandas as pd
# --- GUI related packages
import tqdm


# location = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess1 (only grasp)/'
main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
dataset = '6_real_fall21_x5/'
subfolder = 'PICK/pp1_split/'

location = main + dataset + subfolder

lengths = []

# Divisions to split the time series
divisions = 20

smallest = 10000

for filename in (os.listdir(location)):
    # print(filename)

    # ---- Step 0: Read csv data into an array -----
    # Do not include the first column that has the time, so we don't overfit the next processes
    data = genfromtxt((location + filename), delimiter=',', skip_header=True)

    size = len(data) - 1
    # print('\nThe initial number of point of the time series is: ', size)
    # print('The number of divisions is: ', divisions)

    n_bytes = 5
    points_per_division = math.floor(size / divisions)
    # print('The number of points per division is: ', points_per_division)

    window = math.floor(size / (divisions * n_bytes))
    # print('The window of data_points is: ', window)

    if size < smallest:
        smallest = size

    # --------------- Step 2: Slice data ----------------------------
    for i in range(n_bytes):
        byte = []
        for j in range(divisions):
            start = j * points_per_division + i * window
            end = start + window
            slice = data[start:end]

            # swipe all the columns
            values = []
            for p in range(0, int(slice.size/window)):
                # Start from '1' to avoid including time as a feature or ':' to include it
                value = st.mean(slice[:, p])
                values.append(value)

            # print(slice)
            byte.append(values)

        # Create the new data file
        target_location = main + dataset + 'PICK/pp2_downsampled/'
        name = target_location + filename[:-4] + '_' + str(i) + '.csv'

        with open(name, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerows(byte)