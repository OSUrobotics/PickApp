"""
This file opens all the csvs with the labels of the data that
we collected, and gives some statistics.
"""
import csv
import os
from os.path import exists


# location = '/home/avl/PycharmProjects/AppleProxy/bagfiles/w22_apple_proxy_picks/'       # Lab's Desktop
# location = '/media/avl/StudyData/Apple Pick Data/Apple Proxy Picks/3 - Winter 22 picks/'  # SSD
loc_A = '/home/avl/PycharmProjects/AppleProxy/1_data_valid_for_grasp_and_pick/'
loc_B = '/home/avl/PycharmProjects/AppleProxy/2_data_valid_for_grasp/'


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

                    if rows[0][10] == 's':
                        success = success + 1
                    else:
                        failures = failures + 1

    return success, failures, count


success_A, failures_A, count_A = counter(loc_A)
success_B, failures_B, count_B = counter(loc_B)

success = success_A + success_B
failures = failures_A + failures_B
count = count_A + count_B

print('Success Rate: %.2f ' % (success/(success + failures)))
print(count)
