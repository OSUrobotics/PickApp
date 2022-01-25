"""
This file opens all the csvs with the labels of the data that
we collected, and gives some statistics.
"""
import csv
import os
from os.path import exists

success = 0
failures = 0
count = 0
# location = '/home/avl/PycharmProjects/AppleProxy/bagfiles/w22_apple_proxy_picks/'       # Lab's Desktop
# location = '/media/avl/StudyData/Apple Pick Data/Apple Proxy Picks/3 - Winter 22 picks/'  # SSD

location = '/home/avl/PycharmProjects/AppleProxy/1_data_valid_for_grasp_and_pick/'
for i in range(77):     # The number of real apple picks

    for j in range(13):     # The number of different poses performed at each real-apple pick pose --> adding noise

        file = 'apple_proxy_pick' + str(i) + '-' + str(j) + '_metadata.csv'
        fields = []
        rows = []

        if exists(location + file):
            print(file)
            count += 1

            with open(location + file) as csv_file:
                # Create  a csv object
                csv_reader = csv.reader(csv_file, delimiter=',')
                # Extract field names through first row
                fields = next(csv_reader)
                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)

                if rows[0][10] == 's':
                    success = success + 1
                else:
                    failures = failures + 1

location = '/home/avl/PycharmProjects/AppleProxy/2_data_valid_for_grasp/'
for i in range(77):     # The number of real apple picks

    for j in range(13):     # The number of different poses performed at each real-apple pick pose --> adding noise

        file = 'apple_proxy_pick' + str(i) + '-' + str(j) + '_metadata.csv'
        fields = []
        rows = []

        if exists(location + file):
            print(file)
            count += 1

            with open(location + file) as csv_file:
                # Create  a csv object
                csv_reader = csv.reader(csv_file, delimiter=',')
                # Extract field names through first row
                fields = next(csv_reader)
                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)

                if rows[0][10] == 's':
                    success = success + 1
                else:
                    failures = failures + 1


print('Success Rate: %.2f ' % (success/(success + failures)))
print(count)
