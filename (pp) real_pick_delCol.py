# Math related packages
import numpy as np
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Visualization related packages
from tqdm import tqdm
import matplotlib.pyplot as plt
# Database related packages
import pandas as pd
import os
import csv


# Parameters
experiments = 10
maxdepth = 10


# Autoencoder Features (from pickle Files)
# Location of the Real-Apple data already split and downsampled
# location = 'C:/Users/15416/PycharmProjects\PickApp\data\Real Apples Data/real_data_stage1/'

main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
dataset = '5_real_fall21_x1/'
subfolder = 'pp1_split_x45cols/'

stage = 'PICK/'

location = main + dataset + stage + subfolder
# location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/4_proxy_winter22_x5/PICK/pp1_split_x45cols/'

# First delete the columns
for filename in sorted(os.listdir(location)):
    # print(filename)

    name = str(filename)
    var = name[-5]
    print(var)


    if var == "h":
        # These are WRENCH files, from which we'll delete columns 4 and 8
        print(filename)

        with open(location + filename, "r") as source:
            reader = csv.reader(source)

            new_name = name
            target_location = main + dataset + stage + '/pp1_split/'

            with open(target_location + new_name, "w", newline='') as result:
                writer = csv.writer(result)
                for r in reader:
                    # writer.writerow((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]))
                    writer.writerow((r[0], r[1], r[2], r[3], r[4], r[5], r[6]))

    if var == "u":
        # These are IMU files, from which we'll delete columns 4 and 8
        print(filename)


        with open(location + filename, "r") as source:
            reader = csv.reader(source)

            new_name = name
            target_location = main + dataset + stage + '/pp1_split/'

            with open(target_location + new_name, "w", newline='') as result:
                writer = csv.writer(result)
                for r in reader:
                    # writer.writerow((r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
                    writer.writerow((r[1], r[2], r[3], r[4], r[5], r[6]))

    if var == "s":
        # These are JOIN STATES files, from which we'll delete columns 4 and 8
        print(filename)

        with open(location + filename, "r") as source:
            reader = csv.reader(source)

            new_name = name
            target_location = main + dataset + stage + '/pp1_split/'

            with open(target_location + new_name, "w", newline='') as result:
                writer = csv.writer(result)
                for r in reader:
                    # writer.writerow((r[0], r[1], r[2], r[3]))
                    writer.writerow((r[1], r[2], r[3]))
