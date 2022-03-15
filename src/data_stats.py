"""
This file opens all the csvs that have the labels from the experiments, and returns some statistics
"""
import csv
import os
from os.path import exists


class MetadataStats:

    def __init__(self):

        # Initialize variables and parameters
        self.success = 0
        self.failures = 0
        self.count = 0

    def counter(location):

        # Initialize variables and parameters
        success = 0
        failures = 0
        count = 0

        real_apple_picks = 77   # Number of Real Apple Picks performed before, on which the proxy-pick poses are based
        attempts_at_proxy = 13  # Attempts performed at proxy adding noise to the pose from the real-apple pick

        for i in range(real_apple_picks):

            for j in range(attempts_at_proxy):

                file = 'apple_proxy_pick' + str(i) + '-' + str(j) + '_metadata.csv'
                rows = []

                if exists(location + file):
                    print(file)
                    count += 1

                    with open(location + file) as csv_file:
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


if __name__ == "__main__":

    # Data location
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    dataset = '5_real_fall21_x1/'
    # dataset = '1_proxy_rob537_x1/'
    subfolder = 'pp1_split_x45cols/'

    # location = '/home/avl/PycharmProjects/AppleProxy/bagfiles/w22_apple_proxy_picks/'       # Lab's Desktop
    # location = '/media/avl/StudyData/Apple Pick Data/Apple Proxy Picks/3 - Winter 22 picks/'  # SSD
    # loc_A = '/home/avl/PycharmProjects/AppleProxy/1_data_valid_for_grasp_and_pick/'
    loc_A = 'D:/Apple Pick Data/Apple Proxy Picks/3 - Winter 2022/1_data_valid_for_grasp_and_pick/'  # SSD
    loc_B = 'D:/Apple Pick Data/Apple Proxy Picks/3 - Winter 2022/2_data_valid_for_grasp/'  # SSD


    success_A, failures_A, count_A = counter(loc_A)
    success_B, failures_B, count_B = counter(loc_B)

    success = success_A + success_B
    failures = failures_A + failures_B
    count = count_A + count_B

    print('Success Rate: %.2f ' % (success / (success + failures)))
    print(count)