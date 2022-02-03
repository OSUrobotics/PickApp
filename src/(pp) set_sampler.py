# @Time : 1/26/2022 11:09 AM
# @Author : Alejandro Velasquez

import os
import random
import shutil

# Randomly sample the data into the training set and testing set
labels = ['successful/', 'failed/']
location = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess3 (only grasp_downsampled_ joined)/'

for label in labels:

    for file in os.listdir(location + label):

        coin = random.random()
        print(coin)

        if coin < 0.701:
            # Save file in the training set sub-folder
            target = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess4 (train and test set)/training_set (70%)/'


        else:
            # Save file in the testing set sub-folder
            target = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess4 (train and test set)/testing_set (30%)/'

        original = location + label + file
        target = target + label + file

        shutil.copy(original, target)