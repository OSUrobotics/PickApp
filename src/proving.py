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

    # variable: String

    # Turn csv into pandas - dataframe
    df = pd.read_csv(file)
    # df = np.array(df)

    # print(file)
    # Read the initial readings
    initial_value = df.iloc[1][variable]
    initial_time = df.iloc[0][0]

    initial_value = 0

    # Subtract initial reading to all channels to ease comparison
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
        x_end_idx = tc_idx[1]

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


if __name__ == "__main__":

    # Data Location
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1', '1_proxy_rob537_x1']
    subfolder = '__for_proxy_real_comparison'

    # ----------------------------------------- Step 0 - Variables to choose from --------------------------------------
    variables = [' force_z', ' f1_acc_z', ' f3_acc_z', ' torque_z']
    variable = variables[0]
    # Find the variables's respetive topic
    if variable == ' force_z' or variables == ' force_x' or variables == ' force_y' or variable == ' torque_z':
        topic = 'wrench'
    elif variable == ' f1_acc_x' or variable == ' f1_acc_y' or variable == ' f1_acc_z' or variable == ' f1_gyro_x':
        topic = 'f1_imu'
    elif variable == ' f2_acc_x' or variable == ' f2_acc_y' or variable == ' f2_acc_z' or variable == ' f2_gyro_x':
        topic = 'f2_imu'
    elif variable == ' f3_acc_x' or variable == ' f3_acc_y' or variable == ' f3_acc_z' or variable == ' f3_gyro_x':
        topic = 'f3_imu'

    offset = 10

    # ----------------------------------------- A - SUCCESSFUL PICKS ---------------------------------------------------
    # comparable_picks = [16, 31, 43, 64, 71, 72]
    # case = 'success'
    # for comparable_pick in comparable_picks:
    #
    #     # ------------------------------------- Step 1 - Concatenated Folders ------------------------------------------
    #     proxy_success_pick = str(comparable_pick) + '-10'      # number 10 are the ones without noise added
    #     real_success_pick = comparable_pick
    #     # Build name
    #     real_success_pick_file = 'real_apple_pick_' + str(real_success_pick) + '_pick_' + str(topic) + '.csv'
    #     real_success_grasp_file = 'real_apple_pick_' + str(real_success_pick) + '_grasp_' + str(topic) + '.csv'
    #     proxy_success_pick_file = 'apple_proxy_pick' + str(proxy_success_pick) + '_pick_' + str(topic) + '.csv'
    #     proxy_success_grasp_file = 'apple_proxy_pick' + str(proxy_success_pick) + '_grasp_' + str(topic) + '.csv'
    #     # Concatenate location
    #     real_success_location_pick = main + datasets[1] + '/PICK/' + subfolder + '/' + case + '/' + real_success_pick_file
    #     real_success_location_grasp = main + datasets[1] + '/GRASP/' + subfolder + '/' + case + '/' + real_success_grasp_file
    #     proxy_success_location_pick = main + datasets[0] + '/PICK/' + subfolder + '/' + case + '/' + proxy_success_pick_file
    #     proxy_success_location_grasp = main + datasets[0] + '/GRASP/' + subfolder + '/' + case + '/' + proxy_success_grasp_file
    #
    #     # -------------------------------------------- Step 2 - Bring the data -----------------------------------------
    #     # A - Successful
    #     real_success_pick_time, real_success_pick_value = pic_list(real_success_location_pick, variable)
    #     real_success_grasp_time, real_success_grasp_value = pic_list(real_success_location_grasp, variable)
    #     proxy_success_pick_time, proxy_success_pick_value = pic_list(proxy_success_location_pick, variable)
    #     proxy_success_grasp_time, proxy_success_grasp_value = pic_list(proxy_success_location_grasp, variable)
    #
    #     # ---------------------------------------- Step 4 - Generate array of plots ------------------------------------
    #     f, axrray = plt.subplots(1, 2, figsize=(6, 3), dpi=100, sharey=True)
    #     plt.subplots_adjust(wspace=0.05, hspace=0.175)
    #     # Grasp
    #     ax = axrray[0]
    #     ax.grid()
    #     ax.plot(real_success_grasp_time, real_success_grasp_value, label='Real')
    #     ax.plot(proxy_success_grasp_time, proxy_success_grasp_value, label='Proxy')
    #     ax.legend()
    #     ax.set_ylabel(variable)
    #     y_max = max(max(real_success_pick_value), max(proxy_success_pick_value))
    #     ax.annotate('Grasp', xy=(0, 0.9* y_max), size=15)
    #     # Pick
    #     ax = axrray[1]
    #     ax.grid()
    #     ax.plot(real_success_pick_time, real_success_pick_value, label='Real')
    #     ax.plot(proxy_success_pick_time, proxy_success_pick_value, label='Proxy')
    #     ax.legend()
    #     y_max = max(max(real_success_pick_value), max(proxy_success_pick_value))
    #     ax.annotate('Pick', xy=(0, 0.9 * y_max), size=15)
    #
    #     plt.suptitle('Comparison of successful Real and Proxy pick' + str(comparable_pick), y=1)

    # ----------------------------------------- B - FAILED PICKS -------------------------------------------------------
    comparable_picks = [15, 4]
    case = 'failed'
    for comparable_pick in comparable_picks:

        # ------------------------------------- Step 1 - Concatenated Folders ------------------------------------------
        proxy_failed_pick = str(comparable_pick) + '-10'  # number 10 are the ones without noise added
        real_failed_pick = comparable_pick
        # Build name
        real_failed_pick_file = 'real_apple_pick_' + str(real_failed_pick) + '_pick_' + str(topic) + '.csv'
        real_failed_grasp_file = 'real_apple_pick_' + str(real_failed_pick) + '_grasp_' + str(topic) + '.csv'
        proxy_failed_pick_file = 'apple_proxy_pick' + str(proxy_failed_pick) + '_pick_' + str(topic) + '.csv'
        proxy_failed_grasp_file = 'apple_proxy_pick' + str(proxy_failed_pick) + '_grasp_' + str(topic) + '.csv'
        # Concatenate location
        real_failed_location_pick = main + datasets[1] + '/PICK/' + subfolder + '/' + case + '/' + real_failed_pick_file
        real_failed_location_grasp = main + datasets[1] + '/GRASP/' + subfolder + '/' + case + '/' + real_failed_grasp_file
        proxy_failed_location_pick = main + datasets[0] + '/PICK/' + subfolder + '/' + case + '/' + proxy_failed_pick_file
        proxy_failed_location_grasp = main + datasets[0] + '/GRASP/' + subfolder + '/' + case + '/' + proxy_failed_grasp_file

        # ----------------------------------------- Step 2 - Bring the data --------------------------------------------
        real_failed_pick_time, real_failed_pick_value = pic_list(real_failed_location_pick, variable)
        real_failed_grasp_time, real_failed_grasp_value = pic_list(real_failed_location_grasp, variable)
        proxy_failed_pick_time, proxy_failed_pick_value = pic_list(proxy_failed_location_pick, variable)
        proxy_failed_grasp_time, proxy_failed_grasp_value = pic_list(proxy_failed_location_grasp, variable)

        # ---------------------------------------- Step 4 - Generate array of plots ------------------------------------
        f, axrray = plt.subplots(1, 2, figsize=(6, 3), dpi=100, sharey=True)
        plt.subplots_adjust(wspace=0.05, hspace=0.175)
        # Grasp
        ax = axrray[0]
        ax.grid()
        ax.plot(real_failed_grasp_time, real_failed_grasp_value, label='Real')
        ax.plot(proxy_failed_grasp_time, proxy_failed_grasp_value, label='Proxy')
        ax.legend()
        ax.set_ylabel(variable)
        y_max = max(max(real_failed_pick_value), max(proxy_failed_pick_value))
        ax.annotate('Grasp', xy=(0, 0.9 * y_max), size=15)
        # Pick
        ax = axrray[1]
        ax.grid()
        ax.plot(real_failed_pick_time, real_failed_pick_value, label='Real')
        ax.plot(proxy_failed_pick_time, proxy_failed_pick_value, label='Proxy')
        ax.legend()
        y_max = max(max(real_failed_pick_value), max(proxy_failed_pick_value))
        ax.annotate('Pick', xy=(0, 0.9 * y_max), size=15)

        plt.suptitle('Comparison of Failed Real and Proxy pick' + str(comparable_pick), y=1)


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
