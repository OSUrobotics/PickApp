import pickle
import os
import csv
import statistics as st
import math
import numpy as np
from numpy import genfromtxt
import pandas as pd
from os.path import exists


# Step 1 --- Read the csv file with all the angles from the real-apple picks [Hand-Stem, Stem-Gravity, Hand-gravity]
file = 'data/real_picks_angles_yaw.csv'

# Save it as a list
with open(file, newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

# Convert it into a list of floats, with only the angle of interest
angles = []
for i in data:
    angle = float(i[2])
    angles.append(angle)

print(angles)


# Step 2 --- Read the noise from the metadata files
location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/Apple Proxy Picks/Winter 2022/1_data_valid_for_grasp_and_pick/'
count = 0
hand_gravity_as_feature = []
for i in range(0, len(angles)+1):

    for j in range(13):

        file = 'apple_proxy_pick' + str(i) + '-' + str(j) + '_metadata.csv'
        rows = []

        if exists(location + file):
            print('\n', file)
            count += 1

            with open(location + file) as csv_file:
                # Create  a csv object
                csv_reader = csv.reader(csv_file, delimiter=',')
                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)

                noise = rows[1][16]
                # Convert string representation of list into list
                noise_list = noise.strip('][').split(',')
                print(noise_list)
                roll_noise = float(noise_list[3]) * 180 / 3.1416
                pitch_noise = float(noise_list[4]) * 180 / 3.1416
                yaw_noise = float(noise_list[5]) * 180 / 3.1416

                print('The Hand-Gravity angle is %.2f' % angles[i-1])
                print('The roll noise of this attempt is %.2f:' % roll_noise)
                print('The pitch noise of this attempt is %.2f:' % pitch_noise)

                # Compute the resultant angle
                # TO-DO: Fix the equation
                feature_angle = angles[i-1] + roll_noise + pitch_noise

                # Save list with the name of the pick, and the new feature
                features = [file, feature_angle]

                hand_gravity_as_feature.append(features)

# Step 3 - Create the vector with these features for the Training and Testing Set
print(hand_gravity_as_feature)
# First do it for the training set

# Then for the testing set



# Step 4 - Finally Concatenate it with the Autoencoder features
