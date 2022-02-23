# @Time : 2/15/2022 2:12 PM
# @Author : Alejandro Velasquez

import os
import math
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
from scipy import stats

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
        getattr(x[i * chunk_len : (i + 1) * chunk_len], f_agg)()
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]


def agg_linear_trend(x):
    # Source: https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#agg_linear_trend
    """
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

    :param file:
    :param variable: Given as a strin
    :return: Simplified list
    """

    # ---- Step 1: Turn csv into pandas - dataframe
    df = pd.read_csv(file)
    # df = np.array(df)

    # ---- Step 2: Read the reference readings (whether initial or last) to make the offset
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

    # ---- Step 3: Subtract reference reading to all channels to ease comparison
    # time = df[:, 0] - initial_time
    time = df['# elapsed time'] - initial_time
    # value = df[:, variable] - initial_value
    value = df[variable] - initial_value

    return time, value


def crossings(x, y):

    # Check the initial and ending time of the Force Profile based on the
    # zero crossings

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
        print('Flat')
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

    print('Start at %.2f and ends at %.2f' % (x_init, x_end))
    return x_init, x_end, x_init_idx, x_end_idx


def compare_picks(reals, proxys, topic, main, datasets, subfolder, case, variable):

    distances = []
    best_alignment = 5000       # Start with high value

    for real, proxy in zip(reals, proxys):

        # ------------------------------------- Step 1 - Concatenated Folders ------------------------------------------
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

        # -------------------------------------------- Step 2 - Bring the data -----------------------------------------
        # A - Successful
        real_pick_time, real_pick_value = pic_list(real_location_pick, variable)
        real_grasp_time, real_grasp_value = pic_list(real_location_grasp, variable)
        proxy_pick_time, proxy_pick_value = pic_list(proxy_location_pick, variable)
        proxy_grasp_time, proxy_grasp_value = pic_list(proxy_location_grasp, variable)

        # ---------------------------------------- Step 4 - Dynamic Time Warping ---------------------------------------

        # Temporal Analysis
        a, b, c, d = crossings(real_pick_time, real_pick_value)
        reals = real_pick_value[c:d]
        # plt.plot(reals)

        e, f, g, h = crossings(proxy_pick_time, proxy_pick_value)
        # plt.plot(proxy_pick_value)
        proxys = proxy_pick_value[g:h]
        # plt.plot(proxys)

        # ---- Dynamic Time Warping ----
        alignment = dtw(proxys, reals, keep_internals=True)
        # alignment = dtw(proxy_grasp_value, real_grasp_value, keep_internals=True)

        # if True:
        if alignment.distance < best_alignment:
            best_alignment = alignment.distance
            print(real, proxy, alignment.distance)
            #
            #     # Display the warping curve, i.e. the alignment curve
            #
            alignment.plot(type="alignment")
            alignment.plot(type="threeway")
            alignment.plot(type="twoway", offset=10)
            distances.append(alignment.distance)

            # ---------------------------------------- Step 3 - Generate array of plots ------------------------------------

            f, axrray = plt.subplots(1, 2, figsize=(6, 2), dpi=100, sharey=True)
            plt.subplots_adjust(wspace=0.05, hspace=0.175)

            if variable == ' force_z':
                legend_loc = 'upper right'
            else:
                legend_loc = 'lower right'

            # Grasp
            ax = axrray[0]
            ax.grid()
            ax.plot(real_grasp_time, real_grasp_value, label='Real', color="#de8f05")
            ax.plot(proxy_grasp_time, proxy_grasp_value, label='Proxy', color="#0173b2")
            ax.legend(loc=legend_loc)
            ax.set_ylabel(variable)
            # Location of the Pick and Grasp Labels
            y_max = max(np.max(real_pick_value), np.max(proxy_pick_value))
            if y_max > 1:
                ax.annotate('Grasp', xy=(0, 0.8 * y_max), size=15)
            else:
                ax.annotate('Grasp', xy=(0, -0.8), size=15)

            # Pick
            ax = axrray[1]
            ax.grid()
            ax.plot(real_pick_time, real_pick_value, label='Real', color="#de8f05")
            ax.plot(proxy_pick_time, proxy_pick_value, label='Proxy', color="#0173b2")
            ax.legend(loc=legend_loc)
            # Location of the Pick and Grasp Labels
            y_max = max(np.max(real_pick_value), np.max(proxy_pick_value))
            if y_max > 1:
                ax.annotate('Pick', xy=(0, 0.8 * y_max), size=15)
            else:
                ax.annotate('Pick', xy=(0, -0.8), size=15)

            if case == "success":
                plt.suptitle('Comparison of -- Successful -- Real and Proxy pick' + str(real) + 'vs' + proxy + ' ' + str(alignment.distance), y=1)
            elif case == "failed":
                plt.suptitle('Comparison of -- Failed -- Real and Proxy pick' + str(real) + 'vs' + proxy + ' ' + str(alignment.distance), y=1)


    print(np.mean(distances))


if __name__ == "__main__":

    # Data Location
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1', '1_proxy_rob537_x1']
    subfolder = '__for_proxy_real_comparison'

    # ----------------------------------------- Step 0 - Variables to choose from --------------------------------------
    variables = [' force_z', ' f1_acc_z', ' f3_acc_z', ' torque_z']
    variable = variables[0]
    # Find the variables's respetive topic
    if variable == ' force_z' or variable == ' force_x' or variable == ' force_y' or variable == ' torque_z':
        topic = 'wrench'
    elif variable == ' f1_acc_x' or variable == ' f1_acc_y' or variable == ' f1_acc_z' or variable == ' f1_gyro_x':
        topic = 'f1_imu'
    elif variable == ' f2_acc_x' or variable == ' f2_acc_y' or variable == ' f2_acc_z' or variable == ' f2_gyro_x':
        topic = 'f2_imu'
    elif variable == ' f3_acc_x' or variable == ' f3_acc_y' or variable == ' f3_acc_z' or variable == ' f3_gyro_x':
        topic = 'f3_imu'

    offset = 10

    # ----------------------------------------- A - SUCCESSFUL PICKS ---------------------------------------------------
    case = 'success'
    # Comparable picks
    # 1 - Pairs with the lowest noise
    # real_picks = [10, 16, 30, 31, 43, 50, 51, 53, 60, 64, 6, 70, 71, 72, 73, 74]
    # proxy_picks = ['10-10', '16-12', '30-12', '31-10', '43-10', '50-10', '51-10', '53-7', '60-5', '64-12', '6-6', '70-11', '71-7', '72-0', '73-8', '74-0']
    # Lowest DTW from these pairs
    real_picks = [43]
    proxy_picks = ['43-10']

    # All pairs of real and proxys with same initial pose
    # real_picks = [10, 10, 10, 10, 10, 10, 16, 16, 16, 16, 16, 16, 16, 16, 16, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 53, 60, 60, 60, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 6, 6, 6, 6, 6, 6, 6, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74]
    # proxy_picks = ['10-10', '10-1', '10-1', '10-3', '10-7', '10-9', '16-10', '16-12', '16-2', '16-3', '16-4', '16-5', '16-5', '16-7', '16-9', '30-0', '30-12', '30-1', '30-2', '30-5', '30-6', '30-7', '30-8', '30-9', '31-10', '31-5', '31-6', '31-7', '31-8', '43-0', '43-10', '43-11', '43-12', '43-12', '43-12', '43-4', '43-4', '43-6', '43-7', '43-9', '50-0', '50-10', '50-10', '50-10', '50-3', '51-10', '51-10', '51-10', '51-10', '51-5', '51-5', '53-7', '60-5', '60-6', '60-8', '64-10', '64-11', '64-12', '64-12', '64-2', '64-2', '64-4', '64-4', '64-4', '64-7', '64-8', '64-9', '6-1', '6-1', '6-1', '6-1', '6-5', '6-6', '6-6', '70-11', '70-12', '70-1', '70-2', '70-3', '70-3', '70-5', '70-5', '70-7', '70-9', '71-10', '71-11', '71-11', '71-11', '71-11', '71-4', '71-4', '71-7', '71-8', '71-9', '72-0', '72-10', '72-11', '72-12', '72-12', '72-2', '72-3', '72-4', '72-6', '72-7', '72-8', '72-9', '73-0', '73-10', '73-10', '73-2', '73-3', '73-3', '73-8', '73-9', '74-0', '74-0', '74-12', '74-2', '74-2', '74-2', '74-5', '74-5', '74-5', '74-8', '74-8']
    # Lowest DTW from these pairs
    # real_picks = [43]
    # proxy_picks = ['43-10']

    compare_picks(real_picks, proxy_picks, topic, main, datasets, subfolder, case, variable)


    # # ----------------------------------------- B - FAILED PICKS -----------------------------------------------------
    case = 'failed'
    # Comparable picks

    # 1 - Pairs with the lowest noise
    # real_picks = [12, 13, 14, 15, 17, 18, 19, 1, 26, 27, 28, 29, 2, 32, 35, 39, 44, 45, 47, 49, 4, 54, 55, 56, 5, 68, 69, 8, 9]
    # proxy_picks = ['12-6', '13-5', '14-2', '15-10', '17-9', '18-0', '19-9', '1-1', '26-4', '27-7', '28-1', '29-5', '2-3', '32-1', '35-2', '39-1', '44-1', '45-0', '47-1', '49-2', '4-2', '54-5', '55-11', '56-12', '5-6', '68-6', '69-9', '8-2', '9-4']
    # Lowest DTW from these pairs
    # real_picks = [13]
    # proxy_picks = ['13-5']

    # 2 - All pairs of real and proxys with same initial pose
    # real_picks = [12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 1, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 2, 2, 2, 2, 2, 2, 2, 32, 32, 32, 32, 32, 32, 35, 35, 35, 35, 35, 35, 35, 35, 35, 39, 39, 39, 44, 44, 44, 44, 44, 44, 45, 45, 45, 47, 47, 47, 47, 47, 49, 49, 49, 49, 49, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 5, 5, 5, 5, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
    # proxy_picks = ['12-6', '12-7', '13-0', '13-0', '13-0', '13-0', '13-1', '13-2', '13-2', '13-4', '13-5', '13-6', '13-7', '13-7', '14-2', '14-2', '14-4', '15-10', '15-10', '15-12', '15-1', '15-2', '15-2', '15-2', '15-2', '15-7', '15-8', '17-2', '17-2', '17-2', '17-9', '18-0', '18-0', '18-4', '18-4', '19-5', '19-6', '19-8', '19-9', '1-1', '26-0', '26-0', '26-0', '26-0', '26-0', '26-0', '26-4', '26-4', '26-4', '27-0', '27-1', '27-2', '27-2', '27-2', '27-7', '27-7', '28-1', '28-1', '28-1', '28-4', '29-3', '29-4', '29-5', '29-6', '29-6', '29-6', '29-6', '2-0', '2-1', '2-2', '2-3', '2-3', '2-3', '2-3', '32-0', '32-0', '32-1', '32-2', '32-3', '32-4', '35-0', '35-1', '35-2', '35-3', '35-4', '35-4', '35-4', '35-8', '35-8', '39-1', '39-1', '39-3', '44-1', '44-2', '44-3', '44-4', '44-4', '44-4', '45-0', '45-0', '45-1', '47-1', '47-3', '47-3', '47-3', '47-3', '49-2', '49-3', '49-4', '49-5', '49-5', '4-0', '4-10', '4-1', '4-2', '4-3', '4-4', '4-4', '4-6', '4-7', '4-8', '4-9', '54-11', '54-12', '54-5', '54-6', '54-7', '54-9', '55-11', '55-2', '55-3', '55-4', '55-5', '55-7', '55-8', '56-12', '56-12', '56-12', '56-12', '56-12', '56-12', '5-3', '5-3', '5-6', '5-8', '68-0', '68-0', '68-0', '68-1', '68-2', '68-2', '68-2', '68-6', '68-7', '69-0', '69-11', '69-11', '69-1', '69-1', '69-1', '69-1', '69-1', '69-6', '69-6', '69-9', '8-2', '8-3', '8-4', '8-4', '8-4', '8-4', '9-0', '9-0', '9-1', '9-1', '9-3', '9-4', '9-4', '9-4', '9-4', '9-4']
    # Lowest DTW from these pairs
    # real_picks = [54]
    # proxy_picks = ['54-7']

    # compare_picks(real_picks, proxy_picks, topic, main, datasets, subfolder, case, variable)

    plt.show()



    # -------------------------------------------- Step 3 - Get some features ------------------------------------------
    # Get some common features
    # start, end, start_idx, end_idx = crossings(proxy_pic_time, proxy_pic_values)        # Start and ending time of the force plot
    # print('Picks starts at %.2f and ends at %.2f' %(start, end))
    # agg = agg_linear_trend(proxy_pic_values)
    # agg_located = round(agg_linear_trend(proxy_pic_values[start_idx:end_idx]), 2)


    #
    # plt.figure(figsize=(3.5,4))
    # plt.plot(proxy_pic_time[start_idx-offset:end_idx+offset], proxy_pic_values[start_idx-offset:end_idx+offset], label = "Proxy Pick - ALT = " + str(agg_located))
    # plt.title('Pick Number %s ,  Complete agg is %.3f, and focused agg is %.3f' %(pick_number, agg, agg_located))
    #
    #
    # plt.plot(proxy_pic_time[start_idx-offset:end_idx+offset], proxy_pic_values[start_idx-offset:end_idx+offset], label = "Real Pick - ALT = " + str(agg_located))
    # plt.title('Pick Number %s ,  Complete agg is %.3f, and focused agg is %.3f' %(pick_number, agg, agg_located))
    #
    #
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
