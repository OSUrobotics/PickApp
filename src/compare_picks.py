# @Time : 2/15/2022 2:12 PM
# @Author : Alejandro Velasquez

import csv
import ast
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import linregress, skew
from scipy.signal import cwt, find_peaks_cwt, ricker, welch
from collections import defaultdict
import tqdm
import dtw
from dtw import *
# from dtaidistance import dtw
# from dtaidistance import dtw_visualisation as dtwvis


def number_from_filename(filename):
    """
    Subtracts the number of an apple pick from its filename
    :param filename:
    :return: pick number
    """

    name = str(filename)
    start = name.index('pick')
    end = name.index('meta')
    number = name[start + 4:end - 1]

    return number


def pick_info_from_metadata(location, file, index):
    """
    Extracts the info from a certain index in the metadata file
    :param location: location of the metadata file
    :param file: metadata file
    :param index: index / column where we want to obtain the information from
    :return: information
    """

    rows = []
    with open(location + file) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            rows.append(row)

        info = rows[1][index]

    return info


def same_pose_picks(real_picks_location, proxy_picks_location, label):
    """
    Compares the labels from real and proxy picks and outputs lists of pairs with the same pose and label
    :param real_picks_location: folder with real apple picks
    :param proxy_picks_location: folder with proxy picks
    :param label: whether success or failed picks
    :return: Real and Pick list, where the picks are comparable element-wise
    """

    print("\nFinding real and proxy picks that had the same pose and label...")

    real_list = []
    proxy_list = []

    for file in os.listdir(real_picks_location):

        # Step 1: Get the real-pick number from the filename
        number_real = number_from_filename(file)

        # Step 2: Open metadata and get the label
        real_outcome = pick_info_from_metadata(real_picks_location, file, 10)

        proxy_id = '1'

        for file_prox in os.listdir(proxy_picks_location):

            name = str(file_prox)
            start = name.index('pick')

            # First digit that relates to the real-picks
            end = name.index('-')
            number_proxy = name[start + 4:end]

            # Entire digit
            end = name.index('_m')
            number_proxy_noise = name[start + 4:end]

            # Only open those that have the same number
            if number_proxy == number_real:

                proxy_outcome = pick_info_from_metadata(proxy_picks_location, file_prox, 10)
                proxy_noise = pick_info_from_metadata(proxy_picks_location, file_prox, 16)

                b = ast.literal_eval(proxy_noise)
                c = list(b)

                # Heuristics of the amount of noise
                cart_noise = abs(c[0]) + abs(c[1]) + abs(c[2])
                ang_noise = abs(c[3]) + abs(c[4]) + abs(c[5])

                if real_outcome == proxy_outcome:
                    proxy_id = number_proxy_noise

                    # TODO
                    # print("\nThere is a match:")
                    # print(proxy_outcome)
                    # print(file)
                    # print(file_prox)
                    # print('Cart noise is:', cart_noise)
                    # print('Ang noise is:', ang_noise)

                # NOTE: Unindent twice this if
                if not proxy_id == '1' and real_outcome == label:
                    proxy_list.append(proxy_id)
                    real_list.append(int(number_real))

    # Returns lists of Real and Proxy Picks that had the same pose
    return real_list, proxy_list


def same_pose_lowest_noise_picks(real_picks_location, proxy_picks_location, label):
    """
    Compares the labels from real and proxy picks and outputs lists of pairs with the same pose, same label and lowest
    noise (which represents the closest proxy pick)
    :param real_picks_location: folder with real apple picks
    :param proxy_picks_location: folder with proxy picks
    :param label: whether success or failed picks
    :return: Real and Pick list, where the picks are comparable element-wise
    """

    print("Finding real and proxy picks that had the same pose, label and lowest noise (closes)...")

    real_list = []
    proxy_list = []

    # ---- Same Outcome and pose ----
    for file in os.listdir(real_picks_location):

        # Step 1: Get the real-pick number from the filename
        number_real = number_from_filename(file)

        # Step 2: Open metadata and get the label
        real_outcome = pick_info_from_metadata(real_picks_location, file, 10)

        cart_lowest_noise = 10000
        ang_lowest_noise = 10000

        proxy_id = '1'

        for file_prox in os.listdir(proxy_picks_location):

            name = str(file_prox)
            start = name.index('pick')

            # First digit that relates to the real-picks
            end = name.index('-')
            number_proxy = name[start + 4:end]

            # Entire digit
            end = name.index('_m')
            number_proxy_noise = name[start + 4:end]

            # Only open those that have the same number
            if number_proxy == number_real:

                proxy_outcome = pick_info_from_metadata(proxy_picks_location, file_prox, 10)
                proxy_noise = pick_info_from_metadata(proxy_picks_location, file_prox, 16)

                # print(proxy_noise)
                # Measure the overall noise angular

                b = ast.literal_eval(proxy_noise)
                c = list(b)

                # Heuristics of the amount of noise
                cart_noise = abs(c[0]) + abs(c[1]) + abs(c[2])
                ang_noise = abs(c[3]) + abs(c[4]) + abs(c[5])

                if cart_noise < cart_lowest_noise and ang_noise < ang_lowest_noise and real_outcome == proxy_outcome:
                # if real_outcome == proxy_outcome:
                    cart_lowest_noise = cart_noise
                    ang_lowest_noise = ang_noise
                    proxy_id = number_proxy_noise

                    # TODO
                    # print("\nThere is a match:")
                    # print(proxy_outcome)
                    # print(file)
                    # print(file_prox)
                    # print('Cart noise is:', cart_noise)
                    # print('Ang noise is:', ang_noise)

        if not proxy_id == '1' and real_outcome == label:
            proxy_list.append(proxy_id)
            real_list.append(int(number_real))

    # Returns lists of Real and Proxy Picks that had the same pose
    return real_list, proxy_list


def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len

    :param x: the time series to calculate the aggregation of
    :type x: numpy.ndarray
    :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [
        getattr(x[i * chunk_len: (i + 1) * chunk_len], f_agg)()
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]


def agg_linear_trend(x):
    """
    Source: https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#agg_linear_trend

    Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
    the sequence from 0 up to the number of chunks minus one.
    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.
    The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
    "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.
    The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.
    Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    calculated_agg = defaultdict(dict)
    res_data = []
    res_index = []

    # TODO
    # for parameter_combination in param:
    # print(param)
    # print("\nParam Combi:", parameter_combination)

    # chunk_len = parameter_combination['chunk_len']
    # f_agg = parameter_combination["f_agg"]
    attr = 'rvalue'
    chunk_len = 5
    f_agg = 'mean'

    # attr = 'intercept'
    # chunk_len = 5
    # f_agg = 'min'


    if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
        if chunk_len >= len(x):
            calculated_agg[f_agg][chunk_len] = np.NaN
        else:
            aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
            lin_reg_result = linregress(
                range(len(aggregate_result)), aggregate_result
            )
            calculated_agg[f_agg][chunk_len] = lin_reg_result

    # attr = parameter_combination["attr"]

    if chunk_len >= len(x):
        res_data.append(np.NaN)
    else:
        res_data.append(getattr(calculated_agg[f_agg][chunk_len], attr))

    res_index.append(
        'attr_"{}"__chunk_len_{}__f_agg_"{}"'.format(attr, chunk_len, f_agg)
    )

    # return zip(res_index, res_data)
    return res_data[0]


def pic_list(file, variable):
    """
    Reads a csv file and returns only tha variable of interest and Time.
    This is useful because each topic in ROS saves several channels in one single csv file. Hence we want get only the
    data of the channel that we are interested in.
    :param file: csv file
    :param variable: Given as a string
    :return: Simplified lists (Time list, and values list)
    """

    # Step 1: Turn csv into pandas - dataframe
    df = pd.read_csv(file)
    # df = np.array(df)

    # Step 2: Read the reference readings (whether initial or last) to make the offset
    # Reference value
    if variable == " force_z" or variable == " f1_acc_z":
        if '/GRASP/' in file:
            initial_value = df.iloc[-1][variable]
        if '/PICK/' in file:
            initial_value = df.iloc[0][variable]
    else:
        initial_value = 0
    # Reference time
    initial_time = df.iloc[0][0]

    # Step 3: Subtract reference reading to all channels to ease comparison
    # time = df[:, 0] - initial_time
    time = df['# elapsed time'] - initial_time
    # value = df[:, variable] - initial_value
    value = df[variable] - initial_value

    return time, value


def crossings(x, y):
    """
    Checks the initial and ending time of the Force Profile based on the zero crossings
    :param x: time
    :param y: values
    :return: initial and ending time and its respective indexes
    """

    # --- Step 1: Check the zero crossings of the time-series signal ---
    yy = y - 1  # Small offset to avoid many crossings at zero
    zero_crossings = np.where(np.diff(np.sign(yy)))[0]
    tc = []         # tc: time of crossings
    tc_idx = []
    previous = 0
    for zc in zero_crossings:
        # Only consider crossings apart from each other, otherwise it could be noise
        if x[zc] - previous > 0.25:
            tc.append(x[zc])
            tc_idx.append(zc)
            previous = x[zc]

    # --- Step 2: Select initial point and ending point ----
    if len(tc) == 0:
        # If none were detected, then is flat
        # print('Flat')
        x_init = x[0]
        x_end = x.iloc[-1]
        x_init_idx = 0
        x_end_idx = len(x) - 1

    elif len(tc) >= 2:
        # If two or more crossings are detected, take the first one as the initial, and the second as the final
        x_init = tc[0]
        x_end = tc[1]
        x_init_idx = tc_idx[0]
        x_end_idx = tc_idx[1] + 5

    elif len(tc) == 1:
        # If only one zero-crossing is detected, try increasing the offset value
        x_init = tc[0]
        x_init_idx = tc_idx[0]

        # --- Step 1: Check at what time the slope starts ---
        yy = y - 10  # small offset to avoid many crossings at zero
        zero_crossings = np.where(np.diff(np.sign(yy)))[0]
        tcb = []
        tcb_idx = []
        previous = 0
        for zc in zero_crossings:
            # Only consider crossings apart from each other
            if x[zc] - previous > 0.25:
                tcb.append(x[zc])
                tcb_idx.append(zc)
                previous = x[zc]

        while len(tcb) < 2:
            # print('**************************************************************')
            tcb.append(x_init + 0.5)
            tcb_idx.append(len(x)-1)

        x_end = tcb[1]
        x_end_idx = tcb_idx[1]

    # print('Start at %.2f and ends at %.2f' % (x_init, x_end))
    return x_init, x_end, x_init_idx, x_end_idx


def pick_subplot(axrray, phase, real_times, real_values, proxy_times, proxy_values, variable):
    """
    Creat the subplots of the 'Grasp' and 'Pick' phase of an aple pick
    :param axrray: array of subplots
    :param phase: whether 'Grasp' or 'Pick'
    :param real_times: python list with the time values from real pick
    :param real_values: python list with the variable values from real pick
    :param proxy_times: python list with the time values from proxy pick
    :param proxy_values: python list with the variable values from proxy pick
    :param variable: channel of interest
    :return: none
    """

    if variable == ' force_z':
        legend_loc = 'upper right'
    else:
        legend_loc = 'lower right'

    if phase == 'Grasp':
        position = 0
    else:
        position = 1

    ax = axrray[position]
    ax.grid()
    ax.plot(real_times, real_values, label='Real', color="#de8f05")
    ax.plot(proxy_times, proxy_values, label='Proxy', color="#0173b2")
    ax.legend(loc=legend_loc)

    # Place ylabel only in the left subplot
    if phase == 'Grasp':
        ax.set_ylabel(variable)

    # Location of the Pick and Grasp Labels
    y_max = max(np.max(real_values), np.max(proxy_values))
    if y_max > 1:
        ax.annotate(phase, xy=(0, 0.8 * y_max), size=15)
    else:
        ax.annotate(phase, xy=(0, -0.8), size=15)


def compare_picks(reals, proxys, main, datasets, subfolder, case, variable, phase):
    """
    Compares the apple picks element-wise from the reals and proxys lists
    :param phase: when the dynamic time warping is going to take place
    :param reals: list of picks from real tree
    :param proxys: list of picks from proxy
    :param main: main folder location
    :param datasets: list of real and proxy datasets
    :param subfolder: subfolder location
    :param case: whether successful or failed picks
    :param variable: channel of interest
    :return: none
    """

    dtw_comparison = []
    distances = []
    best_alignment_distance = 5000       # Start with a high value

    topic = topic_from_variable(variable)

    # Compare each pair of picks from real and proxy
    for real, proxy in zip(reals, proxys):

        # --- Step 1 - Concatenate Folders ---
        proxy_pick = proxy
        real_pick = real
        # Build name
        real_pick_file = 'real_apple_pick_' + str(real_pick) + '_pick_' + str(topic) + '.csv'
        real_grasp_file = 'real_apple_pick_' + str(real_pick) + '_grasp_' + str(topic) + '.csv'
        proxy_pick_file = 'apple_proxy_pick' + str(proxy_pick) + '_pick_' + str(topic) + '.csv'
        proxy_grasp_file = 'apple_proxy_pick' + str(proxy_pick) + '_grasp_' + str(topic) + '.csv'
        # Concatenate location
        real_location_pick = main + datasets[1] + '/PICK/' + subfolder + '/' + case + '/' + real_pick_file
        real_location_grasp = main + datasets[1] + '/GRASP/' + subfolder + '/' + case + '/' + real_grasp_file
        proxy_location_pick = main + datasets[0] + '/PICK/' + subfolder + '/' + case + '/' + proxy_pick_file
        proxy_location_grasp = main + datasets[0] + '/GRASP/' + subfolder + '/' + case + '/' + proxy_grasp_file

        # --- Step 2 - Read data ---
        # A - Successful
        real_pick_time, real_pick_value = pic_list(real_location_pick, variable)
        real_grasp_time, real_grasp_value = pic_list(real_location_grasp, variable)
        proxy_pick_time, proxy_pick_value = pic_list(proxy_location_pick, variable)
        proxy_grasp_time, proxy_grasp_value = pic_list(proxy_location_grasp, variable)

        # --- Step 3: Dynamic Time Warping ---
        if phase == 'pick':
            a, b, c, d = crossings(real_pick_time, real_pick_value)
            reals = real_pick_value[c:d]
            e, f, g, h = crossings(proxy_pick_time, proxy_pick_value)
            proxys = proxy_pick_value[g:h]
        elif phase == 'grasp':
            a, b, c, d = crossings(real_grasp_time, real_grasp_value)
            reals = real_grasp_value[c:d]
            e, f, g, h = crossings(proxy_grasp_time, proxy_grasp_value)
            proxys = proxy_grasp_value[g:h]


        try:
            alignment = dtw(proxys, reals, keep_internals=True)
        except IndexError:
            alignment = dtw(10000, 0, keep_internals=True)

        # print(real, proxy, alignment.distance)
        dtw_comparison.append([real, proxy, alignment.distance])
        distances.append(alignment.distance)

        if alignment.distance < best_alignment_distance:
            best_alignment_distance = alignment.distance
            best_alignment = alignment
            best_pair = [real, proxy]
            best_real_grasp_time = real_grasp_time
            best_real_grasp_value = real_grasp_value
            best_proxy_grasp_time = proxy_grasp_time
            best_proxy_grasp_value = proxy_grasp_value
            best_real_pick_time = real_pick_time
            best_real_pick_value = real_pick_value
            best_proxy_pick_time = proxy_pick_time
            best_proxy_pick_value = proxy_pick_value
            # print(real, proxy, alignment.distance)

    # --- Array of Plots (Grasp and Pick) ---
    f, axrray = plt.subplots(1, 2, figsize=(10, 4), dpi=100, sharey=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.175)

    # Plot the grasp phase
    pick_subplot(axrray, 'Grasp',
                 best_real_grasp_time, best_real_grasp_value, best_proxy_grasp_time, best_proxy_grasp_value, variable)
    # Plot pick phase
    pick_subplot(axrray, 'Pick',
                 best_real_pick_time, best_real_pick_value, best_proxy_pick_time, best_proxy_pick_value, variable)

    plt.suptitle('Comparison of (' + case + ') Real pick No.' + str(best_pair[0]) + ' and Proxy pick No.'
                 + best_pair[1] + '\nDynamic Time Warping distance: ' + str(round(best_alignment.distance, 0)), y=1)

    # Save the plot
    name = variable + '__during__' + phase + '.pdf'
    target_dir = os.path.dirname(os.getcwd()) + '/results/'
    f.savefig(target_dir + name)

    # --- Display the best alignment pair ---
    print("The closest pair was:")
    print(best_pair[0], best_pair[1], round(best_alignment.distance,0))
    # TODO
    # best_alignment.plot(type="alignment")
    best_alignment.plot(type="threeway")
    # best_alignment.plot(type="twoway", offset=10)

    # --- Save results in csv ---
    header = ['real', 'proxy', 'dtw']
    name = variable + '__during__' + phase + '.csv'
    target_dir = os.path.dirname(os.getcwd()) + '/results/'
    with open(target_dir + name, 'w') as file:
        write = csv.writer(file)
        write.writerow(header)
        write.writerows(dtw_comparison)


def topic_from_variable(variable):
    """
    Given a variable, it returns the ROS topic associated to it
    :param variable:
    :return: topic
    """

    # Channels associated with each topic
    wrench_variables = [' force_x', ' force_y', ' force_z', ' torque_x', ' torque_y', ' torque_z']
    f1_imu_variables = [' f1_acc_x', ' f1_acc_y', ' f1_acc_z', ' f1_gyro_x', ' f1_gyro_y', ' f1_gyro_z']
    f2_imu_variables = [' f2_acc_x', ' f2_acc_y', ' f2_acc_z', ' f2_gyro_x', ' f2_gyro_y', ' f2_gyro_z']
    f3_imu_variables = [' f3_acc_x', ' f3_acc_y', ' f3_acc_z', ' f3_gyro_x', ' f3_gyro_y', ' f3_gyro_z']

    topic = ''
    if variable in wrench_variables:
        topic = 'wrench'
    elif variable in f1_imu_variables:
        topic = 'f1_imu'
    elif variable in f2_imu_variables:
        topic = 'f2_imu'
    elif variable in f3_imu_variables:
        topic = 'f3_imu'

    return topic


def main():
    """
    This module compares time series from 'real apple' picks and from 'apple proxy'.
    It checks for the closest pair of picks by looking at each of the real apple picks, and sweeping all the
    proxy picks attempts with same pose, and the same label (e.g. successs, failed)
    :return:
    """
    # --- Parse Arguments from Command Line ---
    parser = argparse.ArgumentParser(description='Simple command-line program')
    parser.add_argument('--variable',
                        default='force_z',
                        type=str,
                        help='Channel of interest: "force_x", "force_y", "force_z", "torque_z", "f1_acc_z", "f1_gyro_x"')
    parser.add_argument('--case',
                        default='success',
                        type=str,
                        help='Outcome / label of the apple picks: "success", "failed"')
    parser.add_argument('--phase',
                        default='pick',
                        type=str,
                        help='Phase to do the Dynamic Time Warping analysis: "grasp", "pick"')
    args = parser.parse_args()

    # --- Variable & Topic ---
    variable = ' ' + args.variable
    case = args.case
    phase = args.phase

    # --- Data Location ---
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1']
    subfolder = '/metadata/'

    real_picks_location = main + datasets[1] + subfolder
    proxy_picks_location = main + datasets[0] + subfolder

    # --- Get comparable picks from real and proxy picks
    real_picks, proxy_picks = same_pose_picks(real_picks_location, proxy_picks_location, case[0])

    subfolder = '__for_proxy_real_comparison'
    compare_picks(real_picks, proxy_picks, main, datasets, subfolder, case, variable, phase)

    plt.show()


if __name__ == "__main__":
    main()


    # TODO
    # -------------------------------------------- Step 3 - Get some features ------------------------------------------
    # Get some common features
    # start, end, start_idx, end_idx = crossings(proxy_pic_time, proxy_pic_values)        # Start and ending time of the force plot
    # print('Picks starts at %.2f and ends at %.2f' %(start, end))
    # agg = agg_linear_trend(proxy_pic_values)
    # agg_located = round(agg_linear_trend(proxy_pic_values[start_idx:end_idx]), 2)

    # plt.figure(figsize=(3.5,4))
    # plt.plot(proxy_pic_time[start_idx-offset:end_idx+offset], proxy_pic_values[start_idx-offset:end_idx+offset], label = "Proxy Pick - ALT = " + str(agg_located))
    # plt.title('Pick Number %s ,  Complete agg is %.3f, and focused agg is %.3f' %(pick_number, agg, agg_located))

    # plt.plot(proxy_pic_time[start_idx-offset:end_idx+offset], proxy_pic_values[start_idx-offset:end_idx+offset], label = "Real Pick - ALT = " + str(agg_located))
    # plt.title('Pick Number %s ,  Complete agg is %.3f, and focused agg is %.3f' %(pick_number, agg, agg_located))

    # plt.grid()
    # plt.xlabel("Time [sec]")
    # plt.ylabel("Wrist's Force-z [N]")
    # plt.legend()
    # plt.show()

    # Summary of averages of p-values for different chunk sizes (index=size)
    # Force-z during Pick
    # l = [0.23337979277326423, 0.2203270634907928, 0.18844639895918353, 0.2213198802825746, 0.14204429332042276, 0.1487433443456073, 0.14548226752715077, 0.21360420495781063, 0.08082902666667562, 0.08984952503803083, 0.12448619280869096, 0.10354744851119924, 0.12598206038595966, 0.08338008529129137, 0.11709524606507106, 0.08330071627500696, 0.026952384378935153, 0.04175300449127639, 0.049701897975944154, 0.0974991840161181, 0.15827851813430155, 0.20546669106283139, 0.2616560327521214, 0.11916594244013617, 0.3291276380919355, 0.08625908179047195, 0.22970243878568, 0.09112169585234355, 0.19919647634645105]
    # z-Accel during Grasp
    # l = [0.25901289893444784, 0.2986908716017838, 0.2935827041703743, 0.21092349146729195, 0.2306555221316252, 0.3657076970023651, 0.2132266933123365, 0.25241170080090125, 0.2198693748564147, 0.08689849608604742, 0.3226368691966782, 0.2876482450151785, 0.37029163602415804, 0.6164759317569015, 0.5334846713154942, 0.44816364045643065, 0.44133679900857753, 0.39805566497231737, 0.3587854676400678, 0.19314727740175824, 0.2615276672885519, 0.43612691731668674, 0.1662941779321381, 0.2398107599349763, 0.4174321047067055, 0.27136107580194435, 0.3525662495794601, 0.4030642132974157, 0.4750614196628438]

    # plt.plot(l)
