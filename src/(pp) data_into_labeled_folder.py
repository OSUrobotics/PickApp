# @Time : 2/2/2022 10:05 AM
# @Author : Alejandro Velasquez

import os
import csv
import shutil


main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
# datasets = ['3_proxy_winter22_x1/']
# datasets = ['4_proxy_winter22_x5/']
datasets = ['5_real_fall21_x1/']
# datasets = ['6_real_fall21_x5/']

for dataset in datasets:

    # --- Step 1: Read the metadata folder
    metadata_location = main + dataset + 'metadata/'

    for metadata in (os.listdir(metadata_location)):

        # --- Step 2: Build the name accordingly
        name = str(metadata)

        # Subtract the pick name if needed
        # For the rob-537 picks
        # end = name.index('k')
        # end_2 = name.index('n')
        # name = name[:end + 1] + '_' + name[end + 1:end_2 - 1]

        # For the proxy winter-22 picks
        # end = name.index('k')
        # end_2 = name.index('m')
        # name = name[:end + 1] + '' + name[end + 1:end_2 - 1]

        # For Real-Apple metadata
        start = name.index('r')
        end = name.index('k')
        end_2 = name.index('n')
        name = name[start:end+1] + '_' + name[end + 1:end_2 - 1]

        # --- Step 3: Read the label/result
        rows = []
        with open(metadata_location + metadata) as csv_file:
            # Create  a csv object
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Extract each data row one by one
            for row in csv_reader:
                rows.append(row)
            # Read the label
            if rows[1][10] == 's':
                sub_folder = 'success/'
            else:
                sub_folder = 'failed/'

        # --- Step 4:
        # Spread the files (making a copy)
        # stages = ['GRASP/', 'PICK/']
        stages = ['PICK/']
        # suffixes = ['__grasp', '__pick']
        suffixes = ['_pick_']

        for stage, suffix in zip(stages, suffixes):
            source_location = main + dataset + stage + 'pp1_split/'
            target_location = main + dataset + stage + '__for_proxy_real_comparison/' + sub_folder

            for j in range(5):
                # built_name = name + suffix + '_' + str(j) + '.csv'
                # built_name = name + suffix + str(j) + '.csv'
                built_name = name + suffix + 'wrench.csv'
                # built_name = name + suffix + '.csv'
                shutil.copy(source_location + built_name, target_location + built_name)

        print(name)
        print(sub_folder)

