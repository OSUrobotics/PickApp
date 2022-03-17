"""
This file opens all the csvs that have the labels from the experiments, and returns some basic statistics
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
import matplotlib.pyplot as plt
import seaborn as sns


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

        return success, failures, count

    def get_info(self, column):
        """ Extract values at the column of each metadata file, and concatenate it into a single list
        """
        location = self.main + self.dataset + '/metadata/'

        metadata = []
        for file in os.listdir(location):

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
        Outputs mean, st dev and percentiles of the noise of all the experiments form the data sets.
        It saves the report and boxplots in the results folder.
        :param data:
        :return:
        """

        x_noises = []
        y_noises = []
        z_noises = []
        roll_noises = []
        pitch_noises = []
        yaw_noises = []

        for noise_list in data:
            b = ast.literal_eval(noise_list)
            c = list(b)

            # Cartesian noise
            x_noises.append(c[0])
            y_noises.append(c[1])
            z_noises.append(c[2])

            # Convert angular noise radians to degrees
            roll_noises.append(c[3] * 180 / math.pi)
            pitch_noises.append(c[4] * 180 / math.pi)
            yaw_noises.append(c[5] * 180 / math.pi)

        # --- Boxplots
        fig1, ax1 = plt.subplots()
        ax1.set_title('Cartesian Noise')
        ax1.boxplot([x_noises, y_noises, z_noises])
        plt.xticks([1, 2, 3], ['x_noise', 'y_noise', 'z_noise'])
        name = self.dataset + '__Cartesian_Noise.pdf'
        target_dir = os.path.dirname(os.getcwd()) + '/results/'
        plt.savefig(target_dir + name)

        fig2, ax2 = plt.subplots()
        ax2.set_title('Angular Noise')
        ax2.boxplot([roll_noises, pitch_noises, yaw_noises])
        plt.xticks([1, 2, 3], ['roll_noise', 'pitch_noise', 'yaw_noise'])
        name = self.dataset + '__Angular_Noise.pdf'
        target_dir = os.path.dirname(os.getcwd()) + '/results/'
        plt.savefig(target_dir + name)

        plt.show()

        # --- Print Report
        data = [x_noises, y_noises, z_noises, roll_noises, pitch_noises, yaw_noises]
        labels = ['x_noises', 'y_noises', 'z_noises', 'roll_noises', 'pitch_noises', 'yaw_noises']

        print(" --- Percentiles ---- ")
        for i, j in zip(data, labels):
            print(j)
            print("Mean: ", round(np.mean(i),3), "SD: ", round(np.std(i),3), "Percentiles: ",
              np.percentile(i, [25, 50, 75]))

        # --- Save Report
        name = self.dataset + '__Noise_Report.txt'
        target_dir = os.path.dirname(os.getcwd()) + '/results/'
        with open(target_dir + name, 'w') as file:
            for i, j in zip(data, labels):
                file.write(j + "\n")
                result = "Mean: " + str(round(np.mean(i), 3)) + "SD: " \
                         + str(round(np.std(i), 3)) + "Percentiles: " \
                         + str(np.percentile(i, [25, 50, 75])) + "\n"
                file.write(result)


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

