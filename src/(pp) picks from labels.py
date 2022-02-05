# @Time : 2/4/2022 1:32 PM
# @Author : Alejandro Velasquez

# System related packages
import os
# File related packages
import csv

import pandas as pd


def single_label(location, label_index, label_result):
    """
    Sweeps the files within a folder and outputs a list for each label
    :param label_result:
    :param label_index:
    :param location: Location of csvs with metadata
    """
    pics = []
    total = 0

    for filename in sorted(os.listdir(location)):

        # Make sure to read only the metadata csv files
        if filename.endswith('.csv'):

            total +=1
            rows = []
            with open(location + filename) as csv_file:

                # Create csv object
                csv_reader = csv.reader(csv_file, delimiter=',')

                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)

                # Read the label
                if (rows[1][label_index] == label_result):
                    pics.append(filename)
                    # print(filename)

    print('Quantity: %i out of %i, which is %.2f %%' % (len(pics), total, 100 * len(pics) / total))
    return pics


def and_label(location, label_1_index, label_1_result, label_2_index, label_2_result):
    """
    Sweeps the files within a folder and outputs a list that has both labels
    :param label_2_result:
    :param label_2_index:
    :param label_1_result:
    :param label_1_index:
    :param location: Location of csvs with metadata
    """
    pics = []
    total = 0

    for filename in sorted(os.listdir(location)):

        # Make sure to read only the metadata csv files
        if filename.endswith('.csv'):

            total += 1
            rows = []
            with open(location + filename) as csv_file:

                # Create csv object
                csv_reader = csv.reader(csv_file, delimiter=',')

                # Extract each data row one by one
                for row in csv_reader:
                    rows.append(row)

                # Read the label
                if (rows[1][label_1_index] == label_1_result) and (rows[1][label_2_index] == label_2_result):
                    pics.append(filename)
                    # print(filename)

    print('Quantity: %i out of %i, which is %.2f %%' % (len(pics), total, 100 * len(pics) / total))
    return pics


def pickNumberFromString(string):

    name = str(string)

    if name[0] == 'f':
        # This is the case for those files named "fall21_real_apple_pick13_n..."
        start = name.index('k') + 1
        end = name.index('n') - 1
    elif name[0] == 'a':
        # This is the case for those files named "apple_proxy_pick2-0_metadata"
        start = name.index('k') + 1
        end = name.index('-')

    name = name[start:end]

    return name


def real_vs_failed(real_list, same_result_proxy_list, other_result_proxy_list):

    total_realSuc_proxySuc_count = 0
    total_realSuc_failSuc_count = 0

    for real_suc_pic in real_list:
        # Subtract the real-pic number
        real_pic_number = pickNumberFromString(real_suc_pic)
        # print(real_suc_pic)

        # First check the successful proxy pics
        realSuc_proxySuc_count = 0
        for proxy_suc_pic in same_result_proxy_list:
            # Subtract the proxy-pic number
            proxy_pic_number = pickNumberFromString(proxy_suc_pic)

            if proxy_pic_number == real_pic_number:
                realSuc_proxySuc_count += 1
                total_realSuc_proxySuc_count += 1

        # then check the failed proxy pics
        realSuc_failSuc_count = 0
        for proxy_fail_pic in other_result_proxy_list:
            # Subtract the proxy-pic number
            proxy_pic_number = pickNumberFromString(proxy_fail_pic)

            if proxy_pic_number == real_pic_number:
                realSuc_failSuc_count += 1
                total_realSuc_failSuc_count += 1

        try:
            match = 100 * realSuc_proxySuc_count / (realSuc_proxySuc_count + realSuc_failSuc_count)
            unmatch = 100 * realSuc_failSuc_count / (realSuc_proxySuc_count + realSuc_failSuc_count)
            # print('Successful Real-Pick %s --> Successful Proxy-Picks %.1f%% and Failed Proxy-Pics %.1f%%' % (real_pic_number, match, unmatch))

        except ZeroDivisionError:
            pass
            # print('Success Real Pick %s --> Without valid proxy picks' % real_pic_number)

    total_matches = total_realSuc_proxySuc_count / (total_realSuc_proxySuc_count + total_realSuc_failSuc_count)
    total_unmatches = total_realSuc_failSuc_count / (total_realSuc_proxySuc_count + total_realSuc_failSuc_count)

    # print('After replicating the Successful Real-Picks, %.1f in the proxy were Successful whereas the rest %.1f was not' % (total_matches, total_unmatches))

    return total_matches, total_unmatches

if __name__ == '__main__':
    # --- Step 1: Read the metadata-files and make a list of the successful picks and failure picks

    # Metadata Location
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    # dataset = '1_proxy_rob537_x1/'
    # dataset = '3_proxy_winter22_x1/'

    # Get list of apple-pics that had a drop
    # drop_list = single_label(metadata_loc, 8, 'y')

    # Get list of apple-pics that had a drop and the result was successful
    # drop_and_fail_list = and_label(metadata_loc, 8, 'y', 10, 'f')

    # Get list from real-apple picks that were successful
    dataset = '5_real_fall21_x1/'
    metadata_loc = main + dataset + 'metadata/'
    real_suc_list = single_label(metadata_loc, 10, 's')
    real_fail_list = single_label(metadata_loc, 10, 'f')

    # Get list from apple-proxy picks based which positions were based on the real-apple ones:
    dataset = '3_proxy_winter22_x1/'
    metadata_loc = main + dataset + 'metadata/'
    proxy_suc_list = single_label(metadata_loc, 10, 's')
    proxy_fail_list = single_label(metadata_loc, 10, 'f')

    # Compare Successful real picks with the respective proxy pick:
    suc_matches, suc_unmatches = real_vs_failed(real_suc_list, proxy_suc_list, proxy_fail_list)
    print('After replicating SUCCESSFUL Real-Picks, %.1f in the proxy were'
          ' Successful whereas the rest %.1f was not' % (suc_matches, suc_unmatches))

    # Compare Failed real picks with the respective proxy pick:
    fail_matches, fail_unmatches = real_vs_failed(real_fail_list, proxy_fail_list, proxy_suc_list)
    print('After replicating FAILED Real-Picks, %.1f in the proxy were'
          ' Failed whereas the rest %.1f was not' % (fail_matches, fail_unmatches))