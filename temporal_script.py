
import os
import csv
import statistics as st
import math
import numpy as np
from numpy import genfromtxt
import pandas as pd

location = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess1 (only grasp)/'

lengths = []

# Divisions to split the time series
divisions = 100

smallest = 10000

for filename in os.listdir(location):
    print(filename)

    # ---- Step 0: Read csv data into an array -----
    data = genfromtxt((location + filename), delimiter=',', skip_header=True)
    # print(data)

    size = len(data) - 1
    print('\nThe initial number of point of the time series is: ', size)
    print('The number of divisions is: ', divisions)

    n_bytes = 1
    points_per_division = math.floor(size / divisions)
    print('The number of points per division is: ', points_per_division)

    window = math.floor(size / (divisions * n_bytes))
    print('The window of data_points is: ', window)

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
            for p in range(int(slice.size/window)):
                value = st.mean(slice[:, p])
                values.append(value)

            # print(slice)
            byte.append(values)

        # Create the new data file
        name = 'data_postprocess2 (only grasp _ down sampled)/' + filename[:-4] + '.csv'

        with open(name, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerows(byte)