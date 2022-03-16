"""
This file opens all the csvs that have the labels from the experiments, and returns some statistics
"""

import csv
import os
from os.path import exists
import ast
import math
import numpy as np
import sys
import argparse
from tqdm import tqdm       # Progress Bar Package


class MetadataStats:

    def __init__(self):

        self.main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
        self.dataset = '3_proxy_winter22_x1'
        self.location = self.main + self.dataset + '/metadata/'

    def label_counter(self):
        """
        Counts the success and failures from the metadata files of a single dataset
        """

        location = self.main + self.dataset + '/metadata/'

        # Initialize variables and parameters
        success = 0
        failures = 0
        count = 0

        for filename in os.listdir(location):
            # print(filename)
            rows = []
            with open(location + filename, 'r') as csv_file:  # 'with' closes file automatically afterwards
                # Create  a csv object
                csv_reader = csv.reader(csv_file, delimiter=',')
                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)

                if rows[1][10] == 's':
                    success = success + 1
                else:
                    failures = failures + 1

        # Get the suffix of the filenames, because it varies from one dataset to another
        # for filename in os.listdir(location):
        #     name = str(filename)
        #     break
        # end = name.index('pick')
        # name = name[:end + 4]
        # print(name)

        # real_apple_picks = 77   # Number of Real Apple Picks performed before, on which the proxy-pick poses are based
        # attempts_at_proxy = 13  # Attempts performed at proxy adding noise to the pose from the real-apple pick
        #
        # for i in range(real_apple_picks):
        #
        #     for j in range(attempts_at_proxy):
        #
        #         file = name + str(i) + '-' + str(j) + '_metadata.csv'
        #         rows = []
        #
        #         if exists(location + file):
        #             # print(file)
        #             count += 1
        #
        #             with open(location + file) as csv_file:     # 'with' closes file automatically afterwards
        #                 # Create  a csv object
        #                 csv_reader = csv.reader(csv_file, delimiter=',')
        #                 # Extract each data row one by one
        #                 for row in csv_reader:
        #                     rows.append(row)
        #
        #                 if rows[1][10] == 's':
        #                     success = success + 1
        #                 else:
        #                     failures = failures + 1

        return success, failures, count

    def get_info(self, column):
        """ Extract values at the column of each metadata file, and concatenate it into a single list
        """
        location = self.main + self.dataset + '/metadata/'

        metadata = []
        for file in os.listdir(location):

            # print(file)
            rows = []

            with open(location + file) as csv_file:             # 'with' closes file automatically afterwards

                # Create  a csv object
                csv_reader = csv.reader(csv_file, delimiter=',')

                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)

                # Append data
                metadata.append(rows[1][column])

        return metadata

    def noise_stats(self, data):
        """

        :param data:
        :return:
        """

        x_noises = []
        y_noises = []
        z_noises = []
        r_noises = []
        p_noises = []
        yw_noises = []

        for noise_list in data:
            b = ast.literal_eval(noise_list)
            c = list(b)

            # Cartesian noise
            x_noises.append(c[0])
            y_noises.append(c[1])
            z_noises.append(c[2])

            # Convert angular noise radians to degrees
            r_noises.append(c[3] * 180 / math.pi)
            p_noises.append(c[4] * 180 / math.pi)
            yw_noises.append(c[5] * 180 / math.pi)

        print(" --- Percentiles ---- ")
        print("x noise:")
        print("Mean: ", round(np.mean(x_noises),3), "SD: ", round(np.std(x_noises),3), "Percentiles: ",
              np.percentile(x_noises, [25, 50, 75]))
        print("y noise:")
        print("Mean: ", round(np.mean(y_noises),3), "SD: ", round(np.std(y_noises),3), "Percentiles: ",
              np.percentile(y_noises, [25, 50, 75]))
        print("z noise:")
        print("Mean: ", round(np.mean(z_noises),3), "SD: ", round(np.std(z_noises),3), "Percentiles: ",
              np.percentile(z_noises, [25, 50, 75]))
        print("Roll noise:")
        print("Mean: ", round(np.mean(r_noises),3), "SD: ", round(np.std(r_noises),3), "Percentiles: ",
              np.percentile(r_noises, [25, 50, 75]))
        print("Pitch noise:")
        print("Mean: ", round(np.mean(p_noises),3), "SD: ", round(np.std(p_noises),3), "Percentiles: ",
              np.percentile(p_noises, [25, 50, 75]))
        print("Yaw noise:")
        print("Mean: ", round(np.mean(yw_noises),3), "SD: ", round(np.std(yw_noises),3), "Percentiles: ",
              np.percentile(yw_noises, [25, 50, 75]))

def main():

    # --- Parse Arguments from Command Line ---
    parser = argparse.ArgumentParser(description='Simple command-line program')
    parser.add_argument('--dataset',
                        default='3_proxy_winter22_x1',
                        type=str,
                        help='Select the dataset from: "1_proxy_rob537_x1", "3_proxy_winter22_x1" or "5_real_fall21_x1"')
    args = parser.parse_args()

    # --- Create Object ---
    a = MetadataStats()
    a.dataset = args.dataset

    # --- Call Functions from the Class ---
    success, failures, count = a.label_counter()
    print('Success Rate: %.2f ' % (success / (success + failures)))

    # Calculate functions
    b = a.get_info(16)
    a.noise_stats(b)


if __name__ == "__main__":
    main()

