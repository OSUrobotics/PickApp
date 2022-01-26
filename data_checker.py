"""
Check that all the csv files have their respective data folder
"""

import os

# Data valid for Grasp and Pick
# location = '/home/avl/PycharmProjects/AppleProxy/1_data_valid_for_grasp_and_pick/'

# Data valid for Grasp
# location = '/home/avl/PycharmProjects/AppleProxy/2_data_valid_for_grasp/'

# Data valid for Pick
# location = '/home/avl/PycharmProjects/AppleProxy/3_data_valid_for_pick/'

# Not useful
location = '/home/avl/PycharmProjects/AppleProxy/0_data_not useful/'

location = 'D:/Apple Pick Data/Apple Proxy Picks/3 - Winter 2022/2_data_valid_for_grasp/'              # SSD

all_files = list()
all_dirs = list()

# Step 1 - Get list of csv and subfolders

for item in sorted(os.listdir(location)):

    if item.endswith('.csv'):
        all_files.append(item)

    if os.path.isdir(location + item):
        all_dirs.append(item)

# print('\nThe csv files are: \n', all_files)
# print('\nThe subfolders are: \n', all_dirs)

print('\n\nThere are %i csv files' %len(all_files))
print('\nThere are %i subfolders' %len(all_dirs))


# Step 2 - Compare both folders
if len(all_dirs) <= len(all_files):

    for item in all_files:
        # subtract the final extension to the name
        folder = str(item)
        folder = folder.replace('_metadata.csv', '', 1)
        # print(folder)

        if not os.path.isdir(location + folder):
            print('\n Following folder missing:')
            print(folder)

else:

    for item in all_dirs:
        # subtract the final extension to the name
        file = str(item)
        file = file + '_metadata.csv'
        # print(folder)

        if not os.path.exists(location + file):
            print('\n Following file missing:')
            print(file)




