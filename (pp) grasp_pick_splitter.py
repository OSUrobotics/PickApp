"""
Sweeps all the apple pick bagfiles and saves all the plots from a certain variable in a pdf,
and for a certain label (e.g. Succesful or Failure)
Created by: velasale@oregonstate.edu

References:
https://stackoverflow.com/questions/38938454/python-saving-multiple-subplot-figures-to-pdf
"""

# ... System related packages
import os
import time
# ... File related packages
import pandas as pd
import csv
import bagpy
import scipy.signal
from bagpy import bagreader
# ... Math related packages
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
# ... Plot related packages
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
# ... For Feature Extraction
# from tsfresh import extract_features
# from tsfresh import select_features
# from tsfresh.utilities.dataframe_functions import impute




# sns.set()  # Setting seaborn as default style even if use only matplotlib
plt.close('all')


def filter_variables(variables, parameter):
    """
    This function is meant to filter a list of lists, because usually these built-in functions don't do it
    """
    # Median Filter
    variables_filtered = []
    for i in range(len(variables)):
        variable_filtered = median_filter(variables[i], parameter)
        variables_filtered.append(variable_filtered)

    # Gaussian Filer

    return variables_filtered


def net_value(var_x, var_y, var_z):
    """
    Obtain the net value of a vector, given the 3 components
    :param var_x:
    :param var_y:
    :param var_z:
    :return: net value
    """
    net = [None] * len(var_x)
    for i in range(len(var_x)):
        net[i] = math.sqrt(var_x[i] ** 2 + var_y[i] ** 2 + var_z[i] ** 2)
    return net


def elapsed_time(variable, time_stamp):
    """
    Simplifies the time axis, by subtracting the initial time.
    This is useful because usually the time stamps are given in a long format (i.e. in the order of 1e9)
    :param variable: Reference variable to obtain the size of the time array
    :param time_stamp: The time stamp array that is going to be simplified
    :return: Simplified time as Elapsed Time
    """
    elapsed = [None] * len(variable)
    for i in range(len(variable)):
        elapsed[i] = time_stamp[i] - time_stamp[0]
    return elapsed


def find_instance(array, time_array, threshold, initial_time, case='starts'):
    """
    There are also some events that are important to spot such as when the fingers start moving indeed
    Since these are not published by the hand, we'll find those instances by finding the point at which the slope of any
    of the joints changes
    :param threshold:
    :param array:
    :param time_array:
    :param initial_time: The time at which we start looking for the specific instant.
    :param case: To make this function more general 'starts' is when movement starts, or 'stops' when it stops
    :return: Time at which the variable goes above or below the threshold
    """
    # Step 1: Find the index of the initial time to start looking
    # global i
    for initial_time_index in range(len(time_array)):
        if time_array[initial_time_index] > initial_time:
            break

    # try:
    #     for i in range(initial_time_index, len(array)):
    #         derivative = (array[i + 1] - array[i]) / (time_array[i + 1] - time_array[i])
    #         # print('the derivative is: ', derivative)
    #         if abs(derivative) < rate and case == 'stops':
    #             # print('the derivative is: ', derivative)
    #             # print('The time at which is starts changing is: ', time_array[i])
    #             return i, time_array[i]
    #             break
    #         if abs(derivative) > rate and case == 'starts':
    #             return i, time_array[i]
    #             break

    try:
        for i in range(initial_time_index, len(array)):
            if abs(array[i]) > threshold and case == 'starts':
                break
            elif abs(array[i]) < threshold and case == 'stops':
                break

        return i, time_array[i]

    except KeyError or TypeError:
        if case == 'starts':
            return 1e6, 1e6  # just big numbers
        else:
            return 0, 0  # low numbers


def plot_features(plt, a, b, c, d, e, f, g, h, x_min, x_max, title, label, sigma):
    """
    Add some common features into all the plots in order to ease their analysis, such as shaded areas and dashed lines
    :param x_max:
    :param x_min:
    :param plt:
    :param a: Time of the Open Hand Event
    :param b: Time of the Close Hand Event
    :param c: Time of the Pull Event
    :param d: Time of the Open Hand Event at the end
    :param e: Time of the Servos Start
    :param f: Time of the Servos Stop
    :param g: Time of the UR5 Start
    :param h: Time of the UR5 Stop
    :return:
    """
    plt.legend()
    plt.xlim(x_min, x_max)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axvspan(a, b, color='y', alpha=0.3, lw=0)
    plt.annotate('Open hand', (a, 0.95 * ymin))
    plt.axvspan(b, c, color='b', alpha=0.3, lw=0)
    plt.annotate('Close hand', (b, 0.95 * ymin))
    plt.axvspan(c, d, color='r', alpha=0.3, lw=0)
    plt.annotate('Pull', (c, 0.95 * ymin))
    plt.axvspan(d, xmax, color='g', alpha=0.3, lw=0)
    plt.annotate('Open hand', (d, 0.95 * ymin))
    plt.axvline(x=e, color='k', linestyle='dashed')
    plt.annotate('F1 Servo STARTS moving', (e, 0.85 * ymin))
    plt.axvline(x=f, color='k', linestyle=(0, (5, 10)))
    plt.annotate('F1 Servo STOPS moving', (f, 0.80 * ymin))
    plt.axvline(x=g, color='k', linestyle='dotted')
    plt.annotate('UR5e STARTS moving', (g, 0.85 * ymin))
    plt.axvline(x=h, color='k', linestyle=(0, (1, 10)))
    plt.annotate('UR5e STOPS moving', (h, 0.80 * ymin))
    plt.annotate('sigma = ' + str(sigma), (xmin, ymax))
    plt.xlabel('Elapsed time [sec]')
    plt.title(title + ' ' + f'$\\bf{label}$')
    plt.savefig('plots/' + title + '.png')


def broken_axes(plot_number, axrray, time, variables, legends, e, f, g, h, title, y_label, label, y_lim_max, y_lim_min):
    """
    Creates a plot with two subplots and a break point in the x-axis. This is very useful when a plot is very long and
    you only need to focus on a certain area... in this case in two areas: Grasp and Pick
    Reference: https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
    :param axrray:
    :param plot_number:
    :param y_label:
    :param label: Label of the apple pick: Successful or Failure
    :param time:
    :param variables:
    :param legends:
    :param e:
    :param f:
    :param g:
    :param h:
    :return:
    """

    # # fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', figsize=(16, 9))
    # figu, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w')

    # Just to account the fact that we started numbering in 1
    pos = (plot_number - 1) % 4

    # y_maz is a reference number to place the annotation within the plot
    y_max = []
    for i in range(len(variables)):
        axrray[pos, 0].plot(time, variables[i], label=legends[i])
        axrray[pos, 1].plot(time, variables[i], label=legends[i])
        y_max.append(max(variables[i]))

    ax = axrray[pos, 0]
    ax2 = axrray[pos, 1]

    if pos == 0:
        ax.legend()
        ax2.legend()
        ymax = 1.7 * max(y_max)
        ax.annotate('Grasp', xy=(e, ymax))
        ax2.annotate('Pick', xy=(g, ymax))

    ax.grid()
    ax2.grid()
    ax.set_xlim(e - 0.5, f + 0.5)
    ax2.set_xlim(g - 0.5, h + 0.5)
    ax.set_ylim(y_lim_max, y_lim_min)
    ax2.set_ylim(y_lim_max, y_lim_min)
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    ax.set_ylabel(y_label)
    ax.yaxis.tick_left()
    # ax.tick_params(labelright='off')
    ax2.tick_params(labelleft='off')
    ax2.yaxis.tick_right()

    # d = .015
    d = .02
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    if pos == 3:
        plt.xlabel('Elapsed time [sec]')

    # plt.suptitle(title + ' ' + f'$\\bf{label}$')
    ax.set_title(title + ' ' + f'$\\bf{label}$', size=8, loc='left')
    # # plt.savefig('plots/' + title + ' ' + label + '.png')


def look_at_labels(location, prefix):
    """
    Looks for the csv file of the bagfile, in order to read the labels from the metadata
    :param location:
    :param prefix: the name of the file
    :return:
    """
    # --- Step 1: Look for the file
    for filename in os.listdir(location):
        if filename.startswith(prefix + '_'):
            print(filename)  # print the name of the file to make sure it is what
            break

    # --- Open the file
    with open(location + filename) as f:
        reader = csv.reader(f)
        data = list(reader)
        # print('Result of Pick was: ', data[1][10])

    # --- Redefine label
    if data[1][10] == 's':
        result = '(Successful-Pick)'
    elif data[1][10] == 'f':
        result = '(Failed-Pick)'
    else:
        result = 'heads up... something is wrong'

    return result


def event_times(trial_events_elapsed_time, event_indexes, f1, f2, f3, arm):
    """
    Finds the times at which the hand's servos and arms motors start and stop moving.
    These instants are important because they correspond to the periods when the Grasping and Pick happen.
    Therefore, we would focus on the data within these values, and disregard the rest.
    :param trial_events_elapsed_time:
    :param event_indexes:
    :param f1: Finger 1 [time, speed]
    :param f2: Finger 2 [time, speed]
    :param f3: Finger 3 [time, speed]
    :param arm: Arm Joints' [time, speed]
    :return: All the special instants: open hand, closed hand, move arm, stop arm
    """
    # Initial Open Hand Event
    open_hand_event_index = event_indexes[1]
    open_hand_event_time = trial_events_elapsed_time[open_hand_event_index]

    f1_state_elapsed_time = f1[0]
    f1_state_speed = f1[1]
    f2_state_elapsed_time = f2[0]
    f2_state_speed = f2[1]
    f3_state_elapsed_time = f3[0]
    f3_state_speed = f3[1]

    arm_joints_elapsed_time = arm[0]
    joint_0_spd = arm[1]
    joint_1_spd = arm[2]
    joint_2_spd = arm[3]
    joint_3_spd = arm[4]
    joint_4_spd = arm[5]
    joint_5_spd = arm[6]

    if len(event_indexes) == 2:
        # This was the case of real_apple_pick11
        pulling_apple_event_index = event_indexes[1]
        final_open_hand_event_index = event_indexes[1]
        closing_hand_event_index = event_indexes[1]

    elif len(event_indexes) == 4:
        # This was the case of the real_apple_pick 1 to 10
        pulling_apple_event_index = event_indexes[3]
        final_open_hand_event_index = event_indexes[3]
        closing_hand_event_index = event_indexes[2]

    elif len(event_indexes) == 5:
        # This was the case of the real_apple_pick 12 to 33
        pulling_apple_event_index = event_indexes[3]
        final_open_hand_event_index = event_indexes[4]
        closing_hand_event_index = event_indexes[2]

    elif len(event_indexes) == 6:
        # This was the case of the real_apple_pick 34 to 77
        pulling_apple_event_index = event_indexes[3]
        final_open_hand_event_index = event_indexes[5]
        closing_hand_event_index = event_indexes[2]

    # Be careful when moving from ROS events into hand's, because they don't have the same indexes
    pulling_apple_event_time = trial_events_elapsed_time[pulling_apple_event_index]
    final_open_hand_event_time = trial_events_elapsed_time[final_open_hand_event_index]
    closing_hand_event_time = trial_events_elapsed_time[closing_hand_event_index]

    a = open_hand_event_time
    b = closing_hand_event_time
    c = pulling_apple_event_time
    d = final_open_hand_event_time

    # Servos Start Moving Event
    # Find the instance when the fingers' motors start moving (index and value)
    # print('Point to start evaluating: ', closing_hand_event_time)
    i1, e1 = find_instance(f1_state_speed, f1_state_elapsed_time, 0.01, closing_hand_event_time, 'starts')
    i2, e2 = find_instance(f2_state_speed, f2_state_elapsed_time, 0.01, closing_hand_event_time, 'starts')
    i3, e3 = find_instance(f3_state_speed, f3_state_elapsed_time, 0.01, closing_hand_event_time, 'starts')
    e = min(e1, e2, e3)
    # print('\nFinger servos start moving at: %.2f, %.2f and %.2f ' % (e1, e2, e3))
    # print('The time delay between event and servo moving is: %.2f' % (e - b))

    # Servos Stop Moving Event
    # Find the instance when the finger's motors stop indeed moving
    j1, f1 = find_instance(f1_state_speed, f1_state_elapsed_time, 0.01, e, 'stops')
    j2, f2 = find_instance(f2_state_speed, f2_state_elapsed_time, 0.01, e, 'stops')
    j3, f3 = find_instance(f3_state_speed, f3_state_elapsed_time, 0.01, e, 'stops')
    f = max(f1, f2, f3)
    # print('Finger servos stop moving at: %.2f, %.2f and %.2f' % (f1, f2, f3))

    if len(event_indexes) == 4:
        c = f

    k0, g0 = find_instance(joint_0_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
    k1, g1 = find_instance(joint_1_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
    k2, g2 = find_instance(joint_2_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
    k3, g3 = find_instance(joint_3_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
    k4, g4 = find_instance(joint_4_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
    k5, g5 = find_instance(joint_5_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
    g = min(g0, g1, g2, g3, g4, g5)
    # print(
    #     "The times at which the UR5 joints start are: %.2f, %.2f, %2.f, %2.f, %.2f and %.2f" % (g0, g1, g2, g3, g4, g5))
    # print('\nUR5 starts moving at: %.2f ' % g)

    if len(event_indexes) == 4:
        c = g

    k = max(g0, g1, g2, g3, g4, g5)
    # print("The values of k are: %.2f, %.2f, %2.f, %2.f, %.2f and %.2f" % (k0, k1, k2, k3, k4, k5))

    # Arm Stops pulling apple
    l0, h0 = find_instance(joint_0_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
    l1, h1 = find_instance(joint_1_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
    l2, h2 = find_instance(joint_2_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
    l3, h3 = find_instance(joint_3_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
    l4, h4 = find_instance(joint_4_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
    l5, h5 = find_instance(joint_5_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
    h = max(h0, h1, h2, h3, h4, h5)
    # print(
    #     "The times at which the UR5 joints stop are: %.2f, %.2f, %2.f, %2.f, %.2f and %.2f" % (h0, h1, h2, h3, h4, h5))
    # print('UR5 stops moving at: %.2f' % h)

    return a, b, c, d, e, f, g, h


def grasp_pick_indexes(time_list, grasp_start_time, grasp_end_time, pick_start_time, pick_end_time):

    offset = 0.5

    for i in range(len(time_list)):
        if time_list[i] > (grasp_start_time - offset):
            break

    for j in range(i, len(time_list)):
        if time_list[j] > (grasp_end_time + offset):
            break

    for k in range(j, len(time_list)):
        if time_list[k] > (pick_start_time - offset):
            break

    for l in range(k, len(time_list)):
        if time_list[l] > (pick_end_time + offset):
            break

    return i, j, k, l




def generate_plots(pdf_variable, plot_number, axrray, i, zoom='all', arm_filter_param=3, hand_filter_param=3,
                   target_label='(Successful-Pick)'):
    """
    This is the main function that generates all the plots from a bag file, given the number 'i' of the pick.
    :param pdf_variable:
    :param sigma: Plot filter parameter
    :param zoom: Type of Zoom wanted for the plots... (all, both, grasp or pick)
    :param i: Number of the apple pick
    :return:
    """
    # --------------------------------- Step 1: Search the folder for the bagfiles -------------------------------------
    # Hard Drive
    # location = '/media/avl/StudyData/ApplePicking Data/5 - Real Apple with Hand Closing Fixed/bagfiles'

    # Lab's Laptop
    # location = '/home/avl/PycharmProjects/icra22/bagfiles/'

    # Lab's PC
    # location = '/home/avl/ur_ws/src/apple_proxy/bag_files'
    # location = '/home/avl/PycharmProjects/AppleProxy/bagfiles/'

    # Alejo's laptop
    location = '/home/avl/PycharmProjects/appleProxy/bagfiles/'

    number = str(i)
    # file = 'apple_proxy_pick' + number
    file = 'fall21_real_apple_pick' + number

    # First look at the metadata file, to see if it has the desired label (e.g. success or fail)
    file_label = look_at_labels(location, file)

    if file_label != target_label:
        return 0

    b = bagreader(location + '/' + file + '.bag')

    # Get the list of topics available in the file
    # print(b.topic_table)

    # Read each topic
    # Note: If the csvs from bagfiles are already there, then there is no need to read bagfile, only csv.
    # This is important because it consumes time (specially the ones sampled at 500Hz)
    start_reading_topics = time.time()
    # print('Start reading topics at: ', start_reading_topics)

    csvs_available = True       # TRUE if the csvs have already been read, otherwise FALSE  ... Saves lots of time!

    if csvs_available == False:
        # In this case it has to read the bagfiles and extract the csvs

        # a. Arm's Wrench topic: forces and torques
        data_arm_wrench = b.message_by_topic('wrench')
        arm_wrench = pd.read_csv(data_arm_wrench)

        # b. Arm's Joint_States topic: angular positions of the 6 joints
        data_arm_joints = b.message_by_topic('joint_states')
        arm_joints = pd.read_csv(data_arm_joints)

        # c. Hand's finger1 imu
        data_f1_imu = b.message_by_topic('/applehand/finger1/imu')
        f1_imu = pd.read_csv(data_f1_imu)

        # d. Hand's finger1 joint state
        data_f1_joint = b.message_by_topic('/applehand/finger1/jointstate')
        f1_state = pd.read_csv(data_f1_joint)

        # e. Hand's finger2 imu
        data_f2_imu = b.message_by_topic('/applehand/finger2/imu')
        f2_imu = pd.read_csv(data_f2_imu)

        # f. Hand's finger2 joint state
        data_f2_joint = b.message_by_topic('/applehand/finger2/jointstate')
        f2_state = pd.read_csv(data_f2_joint)

        # g. Hand's finger3 imu
        data_f3_imu = b.message_by_topic('/applehand/finger3/imu')
        f3_imu = pd.read_csv(data_f3_imu)

        # h. Hand's finger3 joint state
        data_f3_joint = b.message_by_topic('/applehand/finger3/jointstate')
        f3_state = pd.read_csv(data_f3_joint)

        # i. Trial events
        data_trial_events = b.message_by_topic('/apple_trial_events')
        trial_events = pd.read_csv(data_trial_events)

    else:
        # a. Arm's Wrench topic: forces and torques
        topic = '/rench.csv'
        csv_from_bag = location + '/' + file + topic
        arm_wrench = pd.read_csv(csv_from_bag)

        # b. Arm's Joint_States topic: angular positions of the 6 joints
        topic = '/oint_states.csv'
        csv_from_bag = location + '/' + file + topic
        arm_joints = pd.read_csv(csv_from_bag)

        # c. Hand's finger1 imu
        topic = '/applehand-finger1-imu.csv'
        csv_from_bag = location + '/' + file + topic
        f1_imu = pd.read_csv(csv_from_bag)

        # d. Hand's finger1 joint state
        topic = '/applehand-finger1-jointstate.csv'
        csv_from_bag = location + '/' + file + topic
        f1_state = pd.read_csv(csv_from_bag)

        # e. Hand's finger2 imu
        topic = '/applehand-finger2-imu.csv'
        csv_from_bag = location + '/' + file + topic
        f2_imu = pd.read_csv(csv_from_bag)

        # f. Hand's finger2 joint state
        topic = '/applehand-finger2-jointstate.csv'
        csv_from_bag = location + '/' + file + topic
        f2_state = pd.read_csv(csv_from_bag)

        # g. Hand's finger3 imu
        topic = '/applehand-finger3-imu.csv'
        csv_from_bag = location + '/' + file + topic
        f3_imu = pd.read_csv(csv_from_bag)

        # h. Hand's finger3 joint state
        topic = '/applehand-finger3-jointstate.csv'
        csv_from_bag = location + '/' + file + topic
        f3_state = pd.read_csv(csv_from_bag)

        # EVENT's topic
        # i. Trial events
        topic = '/apple_trial_events.csv'
        csv_from_bag = location + '/' + file + topic
        trial_events = pd.read_csv(csv_from_bag)

    finish_reading_topics = time.time()
    # print('Finished at: ', finish_reading_topics)
    # print('Elapsed time:', (finish_reading_topics - start_reading_topics))

    # --------------------------  Step 2: Extract each vector from the csv's, and adjust the time-----------------------
    # TIME STAMPS
    arm_time_stamp = arm_wrench.iloc[:, 0]
    arm_joints_time_stamp = arm_joints.iloc[:, 0]
    f1_imu_time_stamp = f1_imu.iloc[:, 0]
    f1_state_time_stamp = f1_state.iloc[:, 0]
    f2_imu_time_stamp = f2_imu.iloc[:, 0]
    f2_state_time_stamp = f2_state.iloc[:, 0]
    f3_imu_time_stamp = f3_imu.iloc[:, 0]
    f3_state_time_stamp = f3_state.iloc[:, 0]
    trial_events_time_stamp = trial_events.iloc[:, 0]

    # ARM
    forces_x = arm_wrench.iloc[:, 5]
    forces_y = arm_wrench.iloc[:, 6]
    forces_z = arm_wrench.iloc[:, 7]
    torques_x = arm_wrench.iloc[:, 8]
    torques_y = arm_wrench.iloc[:, 9]
    torques_z = arm_wrench.iloc[:, 10]

    joint_0_pos = arm_joints.iloc[:, 6]
    joint_1_pos = arm_joints.iloc[:, 7]
    joint_2_pos = arm_joints.iloc[:, 8]
    joint_3_pos = arm_joints.iloc[:, 9]
    joint_4_pos = arm_joints.iloc[:, 10]
    joint_5_pos = arm_joints.iloc[:, 11]

    joint_0_spd = arm_joints.iloc[:, 12]
    joint_1_spd = arm_joints.iloc[:, 13]
    joint_2_spd = arm_joints.iloc[:, 14]
    joint_3_spd = arm_joints.iloc[:, 15]
    joint_4_spd = arm_joints.iloc[:, 16]
    joint_5_spd = arm_joints.iloc[:, 17]

    # HAND
    f1_state_position = f1_state.iloc[:, 5]
    f1_state_speed = f1_state.iloc[:, 6]
    f1_state_effort = f1_state.iloc[:, 7]

    f2_state_position = f2_state.iloc[:, 5]
    f2_state_speed = f2_state.iloc[:, 6]
    f2_state_effort = f2_state.iloc[:, 7]

    f3_state_position = f3_state.iloc[:, 5]
    f3_state_speed = f3_state.iloc[:, 6]
    f3_state_effort = f3_state.iloc[:, 7]

    f1_acc_x = f1_imu.iloc[:, 5]
    f1_acc_y = f1_imu.iloc[:, 6]
    f1_acc_z = f1_imu.iloc[:, 7]

    f2_acc_x = f2_imu.iloc[:, 5]
    f2_acc_y = f2_imu.iloc[:, 6]
    f2_acc_z = f2_imu.iloc[:, 7]

    f3_acc_x = f3_imu.iloc[:, 5]
    f3_acc_y = f3_imu.iloc[:, 6]
    f3_acc_z = f3_imu.iloc[:, 7]

    f1_gyro_x = f1_imu.iloc[:, 8]
    f1_gyro_y = f1_imu.iloc[:, 9]
    f1_gyro_z = f1_imu.iloc[:, 10]

    f2_gyro_x = f2_imu.iloc[:, 8]
    f2_gyro_y = f2_imu.iloc[:, 9]
    f2_gyro_z = f2_imu.iloc[:, 10]

    f3_gyro_x = f3_imu.iloc[:, 8]
    f3_gyro_y = f3_imu.iloc[:, 9]
    f3_gyro_z = f3_imu.iloc[:, 10]

    # Net Values
    net_force = net_value(forces_x, forces_y, forces_z)
    net_torque = net_value(torques_x, torques_y, torques_z)
    net_f1_acc = net_value(f1_acc_x, f1_acc_y, f1_acc_z)
    net_f2_acc = net_value(f2_acc_x, f2_acc_y, f2_acc_z)
    net_f3_acc = net_value(f3_acc_x, f3_acc_y, f3_acc_z)

    arm_elapsed_time = elapsed_time(forces_x, arm_time_stamp)
    arm_joints_elapsed_time = elapsed_time(arm_joints, arm_joints_time_stamp)
    f1_imu_elapsed_time = elapsed_time(f1_imu, f1_imu_time_stamp)
    f1_state_elapsed_time = elapsed_time(f1_state, f1_state_time_stamp)
    f2_imu_elapsed_time = elapsed_time(f2_imu, f2_imu_time_stamp)
    f2_state_elapsed_time = elapsed_time(f2_state, f2_state_time_stamp)
    f3_imu_elapsed_time = elapsed_time(f3_imu, f3_imu_time_stamp)
    f3_state_elapsed_time = elapsed_time(f3_state, f3_state_time_stamp)
    trial_events_elapsed_time = elapsed_time(trial_events, trial_events_time_stamp)

    # ----------------------------- Step 3: Get the Events' times ------------------------------------------------------
    # First get the indexes when the events happen
    event_indexes = np.where(np.diff(trial_events.iloc[:, 1], prepend=np.nan))[0]
    # print('The events indexes are: ', event_indexes)

    a, b, c, d, e, f, g, h = event_times(trial_events_elapsed_time,
                                         event_indexes,
                                         [f1_state_elapsed_time, f1_state_speed],
                                         [f2_state_elapsed_time, f2_state_speed],
                                         [f3_state_elapsed_time, f3_state_speed],
                                         [arm_joints_elapsed_time, joint_0_spd, joint_1_spd, joint_2_spd, joint_3_spd,
                                          joint_4_spd, joint_5_spd])

    x_min = min(arm_time_stamp)
    x_max = max(arm_time_stamp)

    if zoom == 'all':
        x_min = 0
        x_max = max(arm_elapsed_time)
    elif zoom == 'grasp':
        x_min = e - 0.8
        x_max = f + 1
    elif zoom == 'pick':
        x_min = g - 1
        x_max = h + 1

    # --------------------------------------  Step 4: Filter Data  -----------------------------------------------------
    # Smooth data

    # Unify variables in a single list
    arm_variables = [forces_x, forces_y, forces_z, net_force,               # 0, 1, 2, 3
                     torques_x, torques_y, torques_z, net_torque,           # 4, 5, 6, 7
                     joint_0_pos, joint_1_pos, joint_2_pos, joint_3_pos, joint_4_pos, joint_5_pos]  # 8, 9, 10, 11, 12, 13

    hand_variables = [f1_acc_x, f1_acc_y, f1_acc_z, net_f1_acc,             # 0, 1, 2, 3,
                      f2_acc_x, f2_acc_y, f2_acc_z, net_f2_acc,             # 4, 5, 6, 7,
                      f3_acc_x, f3_acc_y, f3_acc_z, net_f3_acc,             # 8, 9, 10, 11,
                      f1_gyro_x, f1_gyro_y, f1_gyro_z,                      # 12, 13, 14,
                      f2_gyro_x, f2_gyro_y, f2_gyro_z,                      # 15, 16, 17,
                      f3_gyro_x, f3_gyro_y, f3_gyro_z,                      # 18, 19, 20,
                      f1_state_position, f1_state_speed, f1_state_effort,   # 21, 22, 23,
                      f2_state_position, f2_state_speed, f2_state_effort,   # 24, 25, 26,
                      f3_state_position, f3_state_speed, f3_state_effort]   # 27, 28, 29,

    arm_labels = ['forces_x', 'forces_y', 'forces_z', 'net_force',
                  'torques_x', 'torques_y', 'torques_z', 'net_torque',
                  'joint_0_pos', 'joint_1_pos', 'joint_2_pos', 'joint_3_pos', 'joint_4_pos', 'joint_5_pos']  # 8, 9, 10, 11, 12, 13

    hand_labels = ['f1_acc_x', 'f1_acc_y', 'f1_acc_z', 'net_f1_acc',            # 0, 1, 2, 3
                   'f2_acc_x', 'f2_acc_y', 'f2_acc_z', 'net_f2_acc',            # 4, 5, 6, 7
                   'f3_acc_x', 'f3_acc_y', 'f3_acc_z', 'net_f3_acc',            # 8, 9, 10, 11
                   'f1_gyro_x', 'f1_gyro_y', 'f1_gyro_z',
                   'f2_gyro_x', 'f2_gyro_y', 'f2_gyro_z',
                   'f3_gyro_x', 'f3_gyro_y', 'f3_gyro_z',
                   'f1_state_position', 'f1_state_speed', 'f1_state_effort',
                   'f2_state_position', 'f2_state_speed', 'f2_state_effort',
                   'f3_state_position', 'f3_state_speed', 'f3_state_effort']

    # Apply filter to all the variables
    arm_variables_filtered = filter_variables(arm_variables, arm_filter_param)
    hand_variables_filtered = filter_variables(hand_variables, hand_filter_param)

    # ------------------------------------ Step 5: Crop Data for csv's -------------------------------------------------
    # (b) Human Readable Time Stamps
    arm_time_ref = arm_elapsed_time
    arm_joints_time_ref = arm_joints_elapsed_time
    f1_state_time_ref = f1_state_elapsed_time
    f1_imu_time_ref = f1_imu_elapsed_time
    f2_state_time_ref = f2_state_elapsed_time
    f2_imu_time_ref = f2_imu_elapsed_time
    f3_state_time_ref = f3_state_elapsed_time
    f3_imu_time_ref = f3_imu_elapsed_time
    trial_events_time_ref = trial_events_elapsed_time

    # --- ARM WRENCH ---
    # Get the indexes of the grasp and pick moments, in order to obtain the slices
    g_on, g_off, p_on, p_off = grasp_pick_indexes(arm_time_ref, e, f, g, h)
    header = 'elapsed time, force_x, force_y, force_z, net_force, torque_x, torque_y, torque_z, net_torque'
    # Grasp
    arm_list = [arm_time_ref[g_on:g_off],
                arm_variables_filtered[0][g_on:g_off],              # Slice the list only during the grasp
                arm_variables_filtered[1][g_on:g_off],              # Slice the list only during the grasp
                arm_variables_filtered[2][g_on:g_off],              # Slice the list only during the grasp
                arm_variables_filtered[3][g_on:g_off],
                arm_variables_filtered[4][g_on:g_off],
                arm_variables_filtered[5][g_on:g_off],
                arm_variables_filtered[6][g_on:g_off],
                arm_variables_filtered[7][g_on:g_off]
                ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_grasp_wrench.csv", np.array(arm_list).T, delimiter=',', header=header)
    # Pick
    arm_list = [arm_time_ref[p_on:p_off],                           # Slice the list only during the grasp
                arm_variables_filtered[0][p_on:p_off],              # Slice the list only during the grasp
                arm_variables_filtered[1][p_on:p_off],              # Slice the list only during the grasp
                arm_variables_filtered[2][p_on:p_off],              # Slice the list only during the grasp
                arm_variables_filtered[3][p_on:p_off],
                arm_variables_filtered[4][p_on:p_off],
                arm_variables_filtered[5][p_on:p_off],
                arm_variables_filtered[6][p_on:p_off],
                arm_variables_filtered[7][p_on:p_off]
                ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_pick_wrench.csv", np.array(arm_list).T, delimiter=',', header=header)

    # --- HAND FINGER 1 ---
    # Get the indexes of the grasp and pick moments, in order to obtain the slices
    g_on, g_off, p_on, p_off = grasp_pick_indexes(f1_state_time_ref, e, f, g, h)
    header = 'elapsed_time, f1_state_position, f1_state_speed, f1_state_effort'
    # Grasp
    f1_states_list = [f1_state_time_ref[g_on:g_off],
                      hand_variables_filtered[21][g_on:g_off],
                      hand_variables_filtered[22][g_on:g_off],
                      hand_variables_filtered[23][g_on:g_off]
                      ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_grasp_f1_states.csv", np.array(f1_states_list).T, delimiter=',', header=header)
    # Pick
    f1_states_list = [f1_state_time_ref[p_on:p_off],
                      hand_variables_filtered[21][p_on:p_off],
                      hand_variables_filtered[22][p_on:p_off],
                      hand_variables_filtered[23][p_on:p_off]
                      ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_pick_f1_states.csv", np.array(f1_states_list).T, delimiter=',', header=header)

    # Grasp
    # Get the indexes of the grasp and pick moments, in order to obtain the slices
    g_on, g_off, p_on, p_off = grasp_pick_indexes(f1_imu_time_ref, e, f, g, h)
    header = 'elapsed time, f1_acc_x, f1_acc_y, f1_acc_z, f1_acc_net, f1_gyro_x, f1_gyro_y, f1_gyro_z'
    # Grasp
    f1_imu_list = [f1_imu_time_ref[g_on:g_off],
                      hand_variables_filtered[0][g_on:g_off],
                      hand_variables_filtered[1][g_on:g_off],
                      hand_variables_filtered[2][g_on:g_off],
                      hand_variables_filtered[3][g_on:g_off],
                      hand_variables_filtered[12][g_on:g_off],
                      hand_variables_filtered[13][g_on:g_off],
                      hand_variables_filtered[14][g_on:g_off],
                      ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_grasp_f1_imu.csv", np.array(f1_imu_list).T, delimiter=',',
               header=header)
    # Pick
    f1_imu_list = [f1_imu_time_ref[p_on:p_off],
                      hand_variables_filtered[0][p_on:p_off],
                      hand_variables_filtered[1][p_on:p_off],
                      hand_variables_filtered[2][p_on:p_off],
                      hand_variables_filtered[3][p_on:p_off],
                      hand_variables_filtered[12][p_on:p_off],
                      hand_variables_filtered[13][p_on:p_off],
                      hand_variables_filtered[14][p_on:p_off],
                      ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_pick_f1_imu.csv", np.array(f1_imu_list).T, delimiter=',',
               header=header)



    # --- HAND FINGER 2 ---
    # Get the indexes of the grasp and pick moments, in order to obtain the slices
    g_on, g_off, p_on, p_off = grasp_pick_indexes(f2_state_time_ref, e, f, g, h)
    header = 'elapsed_time, f2_state_position, f2_state_speed, f2_state_effort'
    # Grasp
    f2_states_list = [f2_state_time_ref[g_on:g_off],
                      hand_variables_filtered[24][g_on:g_off],
                      hand_variables_filtered[25][g_on:g_off],
                      hand_variables_filtered[26][g_on:g_off]
                      ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_grasp_f2_states.csv", np.array(f2_states_list).T, delimiter=',',
               header=header)
    # Pick
    f2_states_list = [f2_state_time_ref[p_on:p_off],
                      hand_variables_filtered[24][p_on:p_off],
                      hand_variables_filtered[25][p_on:p_off],
                      hand_variables_filtered[26][p_on:p_off]
                      ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_pick_f2_states.csv", np.array(f2_states_list).T, delimiter=',',
               header=header)

    # Grasp
    # Get the indexes of the grasp and pick moments, in order to obtain the slices
    g_on, g_off, p_on, p_off = grasp_pick_indexes(f2_imu_time_ref, e, f, g, h)
    header = 'elapsed time, f2_acc_x, f2_acc_y, f2_acc_z, f2_acc_net, f2_gyro_x, f2_gyro_y, f2_gyro_z'
    # Grasp
    f2_imu_list = [f2_imu_time_ref[g_on:g_off],
                   hand_variables_filtered[4][g_on:g_off],
                   hand_variables_filtered[5][g_on:g_off],
                   hand_variables_filtered[6][g_on:g_off],
                   hand_variables_filtered[7][g_on:g_off],
                   hand_variables_filtered[15][g_on:g_off],
                   hand_variables_filtered[16][g_on:g_off],
                   hand_variables_filtered[17][g_on:g_off],
                   ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_grasp_f2_imu.csv", np.array(f2_imu_list).T, delimiter=',',
               header=header)
    # Pick
    f2_imu_list = [f2_imu_time_ref[p_on:p_off],
                   hand_variables_filtered[4][p_on:p_off],
                   hand_variables_filtered[5][p_on:p_off],
                   hand_variables_filtered[6][p_on:p_off],
                   hand_variables_filtered[7][p_on:p_off],
                   hand_variables_filtered[15][p_on:p_off],
                   hand_variables_filtered[16][p_on:p_off],
                   hand_variables_filtered[17][p_on:p_off],
                   ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_pick_f2_imu.csv", np.array(f2_imu_list).T, delimiter=',',
               header=header)


    # --- HAND FINGER 3 ---
    # Get the indexes of the grasp and pick moments, in order to obtain the slices
    g_on, g_off, p_on, p_off = grasp_pick_indexes(f3_state_time_ref, e, f, g, h)
    header = 'elapsed_time, f3_state_position, f3_state_speed, f3_state_effort'
    # Grasp
    f3_states_list = [f3_state_time_ref[g_on:g_off],
                      hand_variables_filtered[27][g_on:g_off],
                      hand_variables_filtered[28][g_on:g_off],
                      hand_variables_filtered[29][g_on:g_off]
                      ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_grasp_f3_states.csv", np.array(f3_states_list).T, delimiter=',',
               header=header)
    # Pick
    f3_states_list = [f3_state_time_ref[p_on:p_off],
                      hand_variables_filtered[27][p_on:p_off],
                      hand_variables_filtered[28][p_on:p_off],
                      hand_variables_filtered[29][p_on:p_off]
                      ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_pick_f3_states.csv", np.array(f3_states_list).T, delimiter=',',
               header=header)

    # Grasp
    # Get the indexes of the grasp and pick moments, in order to obtain the slices
    g_on, g_off, p_on, p_off = grasp_pick_indexes(f3_imu_time_ref, e, f, g, h)
    header = 'elapsed time, f3_acc_x, f3_acc_y, f3_acc_z, f3_acc_net, f3_gyro_x, f3_gyro_y, f3_gyro_z'
    # Grasp
    f3_imu_list = [f3_imu_time_ref[g_on:g_off],
                   hand_variables_filtered[8][g_on:g_off],
                   hand_variables_filtered[9][g_on:g_off],
                   hand_variables_filtered[10][g_on:g_off],
                   hand_variables_filtered[11][g_on:g_off],
                   hand_variables_filtered[18][g_on:g_off],
                   hand_variables_filtered[19][g_on:g_off],
                   hand_variables_filtered[20][g_on:g_off],
                   ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_grasp_f3_imu.csv", np.array(f3_imu_list).T, delimiter=',',
               header=header)
    # Pick
    f3_imu_list = [f3_imu_time_ref[p_on:p_off],
                   hand_variables_filtered[8][p_on:p_off],
                   hand_variables_filtered[9][p_on:p_off],
                   hand_variables_filtered[10][p_on:p_off],
                   hand_variables_filtered[11][p_on:p_off],
                   hand_variables_filtered[18][p_on:p_off],
                   hand_variables_filtered[19][p_on:p_off],
                   hand_variables_filtered[20][p_on:p_off],
                   ]
    np.savetxt("csvs/real_apple_pick_" + str(i) + "_pick_f3_imu.csv", np.array(f3_imu_list).T, delimiter=',',
               header=header)


    return 1


# List of all the variables that you want to generate the pdfs
variables = ['1 - UR5e - (FT) Wrist Forces',
             # '2 - UR5e - (FT) Wrist Torques',
             # '3 - UR5e - Joint Angular Positions',
             # '4 - Finger 1 - Joint States',
             # '5 - Finger 1 - (IMU) Linear Acceleration',
             # '6 - Finger 1 - (IMU) Angular Velocity',
             # '7 - Finger 2 - Joint States',
             # '8 - Finger 2 - (IMU) Linear Acceleration',
             # '9 - Finger 2 - (IMU) Angular Velocity',
             # '10 - Finger 3 - Joint States',
             # '11 - Finger 3 - (IMU) Linear Acceleration',
             # '12 - Finger 3 - (IMU) Angular Velocity'
             ]

# Sigma value to filter the plot data
hand_filter_window = 3  # Have in mind that the Sampling Rate of the Hand was 70Hz
arm_filter_window = int(hand_filter_window * 500 / 70)  # Have in mind that the Sampling Rate of the Arm was 500Hz
# arm_filter_window = 1

# Label that we are looking for. Note: It has to be same as in function def look_at_labels
label = '(Successful-Pick)'
# label = '(Failed-Pick)'

for pdf_variable in variables:

    # # Step 1: Create PDF
    # pdf_pages = PdfPages(str(pdf_variable) + ' ' + label + ' (Median Filter - Arm = ' +
    #                      "{:.0f}".format(arm_filter_window) + ', Hand = ' + str(hand_filter_window) + ').pdf')

    # # Step 2: Create the plot array
    nb_plots = 77
    nb_plots_per_page = 4
    f, axrray = plt.subplots(nb_plots_per_page, 2, figsize=(8.5, 14), dpi=100, sharey=True)
    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.175)
    plot_number = 1

    # # Step 3: Generate all the plots for that particular variable within all the picks
    n_picks = 77


    for pick in range(1, n_picks + 1):

        try:
            # Skip the picks that had some issues
            if pick in [11, 36, 37, 38, 40, 41, 42]:
                continue

            print("\nGenerating plots for pick No.: ", pick)

            # Plot stuff
            toss = generate_plots(pdf_variable, plot_number, axrray, pick, 'pick', arm_filter_window,
                                  hand_filter_window, label)  # type 'all', 'grasp' or 'pick'

            # print('Toss and Plot number are:', toss, plot_number)  # toss=0 if it is not the desired label

            # plot_number = plot_number + toss
            #
            # # Create a new page everytime the number of rows is reached
            # if (toss == 1 and (plot_number - 1) % nb_plots_per_page == 0) or (
            #         plot_number - 1) == nb_plots or pick == n_picks:
            #     print('Page saved\n')
            #     pdf_pages.savefig(f)
            #     f, axrray = plt.subplots(nb_plots_per_page, 2, figsize=(8.5, 14), dpi=100, sharey=True)
            #     # Margins
            #     plt.subplots_adjust(left=0.075, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.175)

        except KeyboardInterrupt:
            # # You may cancel the process at any time, and the plots will still be saved
            # pdf_pages.savefig(f)
            print('User cancelled the process.')
            # pdf_pages.close()

    # print('PDF saved')
    # pdf_pages.close()
