# @Time : 4/7/2022 11:15 AM
# @Author : Alejandro Velasquez

import os
import numpy as np
from numpy import genfromtxt
import math
import statistics as st

import pandas as pd


def downsample(source):

    size_target = 45    # this was the fewest sampled points

    for filename in (os.listdir(source)):
        print(filename)

        # --- Step 0: Read csv data into a a Pandas Dataframe ---
        # Do not include the first column that has the time, so we don't overfit the next processes
        # data = genfromtxt((source + filename), delimiter=',', skip_header=True)

        data = pd.read_csv(source + filename)
        n_samples = data.shape[0]       # rows
        n_channels = data.shape[1]      # columns

        samples_per_point = math.floor(n_samples / size_target)

        for i in range(n_channels):

            for j in range(size_target):

                start = samples_per_point * j
                end = start + samples_per_point

                print(data.iloc[start:end, i])





        values = []
        for i in range(size_target):
            start = samples_per_point * i
            end = start + samples_per_point
            slice = data[start:end]
            print(slice)


    print(smallest)



def main():

    a = 1
    # Step 1 - Read Data saved as csvs from bagfiles

    # Step 2 - Split the data into Grasp and Pick
    # (pp) grasp_and_pick_split.py

    # Step 3 - Select the columns to pick
    # (pp) real_pick_delCol.py

    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    dataset = '3_proxy_winter22_x1/'
    stages = ['GRASP/', 'PICK/']

    # --- Step 4 - Down sample Data ---
    # (pp) downsampler.py
    for stage in stages:
        source_location = main + dataset + stage + 'pp1_split/'
        print(source_location)

        downsample(source_location)



    # Step 5 - Join all topics into a single csv.
    # (pp) csv_joiner.py

    # Step 6 - Do Data Augmentation by adding Noise
    # TODO

    # Step 7 - Save csvs in subfolders labeled
    # (pp) data_into_labeled_folder.py


if __name__ == '__main__':
    main()