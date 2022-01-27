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
from os.path import exists
from tqdm import tqdm
# ... File related packages
import pandas as pd
import csv
import bagpy
from bagpy import bagreader
# ... Math related packages
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
# ... Plot related packages
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# sns.set()  # Setting seaborn as default style even if use only matplotlib
plt.close('all')


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

    # y_max is a reference number to know where to place the annotation within the plot
    y_max = []
    for i in range(len(variables)):
        axrray[pos, 0].plot(time, variables[i], label=legends[i])
        axrray[pos, 1].plot(time, variables[i], label=legends[i])
        y_max.append(max(variables[i]))

    ax = axrray[pos, 0]
    ax2 = axrray[pos, 1]

    # Place the labels 'Grasp' and 'Pick' at the top of the pdf page
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

    # Place the label 'Time' at the bottom of the pdf page
    if pos == 3:
        plt.xlabel('Elapsed time [sec]')

    # plt.suptitle(title + ' ' + f'$\\bf{label}$')
    ax.set_title(title + ' ' + f'$\\bf{label}$', size=8, loc='left')
    # # plt.savefig('plots/' + title + ' ' + label + '.png')


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

    elif len(event_indexes) == 3:
        # This is for the last improvement
        pulling_apple_event_index = event_indexes[0]
        final_open_hand_event_index = event_indexes[1]
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

    # A simpler approach: just by using the states.

    # Grasp Events
    start_grasp_apple_event_index = event_indexes[0]
    e = trial_events_elapsed_time[start_grasp_apple_event_index]
    f = e + 2.0

    # Pick Events
    start_pulling_apple_event_index = event_indexes[1]
    finish_pulling_apple_event_index = event_indexes[2]
    g = trial_events_elapsed_time[start_pulling_apple_event_index]
    h = trial_events_elapsed_time[finish_pulling_apple_event_index]

    return a, b, c, d, e, f, g, h


def filter_variables(variables, parameter):
    """
    This function is meant to filter a list of lists, because usually these built-in functions don't do it
    """
    # Median Filter
    variables_filtered = []
    for i in range(len(variables)):
        variable_filtered = median_filter(variables[i], parameter)
        variables_filtered.append(variable_filtered)

    return variables_filtered


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


def generate_plots(location, bagfile, pdf_variable, plot_number, axrray, arm_filter_param=3, hand_filter_param=3):
    """
    This is the main function that generates all the plots from a bag file, given the number 'i' of the pick.
    :param pdf_variable: The variable for which we are creating the pdf, so we can compare all the plots easily
    :param plot_number: Useful to know in what place in the page of the pdf to put this plot
    :param axrray: Array of subplots
    :param i: Apple pick number
    :param arm_filter_param: Filter parameter used for the Arm's signals
    :param hand_filter_param: Filter parameter used for the Hand's signals
    :return:
    """


    file = str(bagfile)
    file = file.replace('.bag', '', 1)

    # Get the list of topics available in the file
    # print(b.topic_table)

    # --- Read each topic ---
    # Note: If the csvs from bagfiles are already there, then there is no need to read bagfile, only csv.
    # This is important because it consumes time (specially the ones sampled at 500Hz)
    start_reading_topics = time.time()
    # print('Start reading topics at: ', start_reading_topics)

    try:
        # First try if the csvs are already available
        # a. Arm's Wrench topic: forces and torques
        topic = '/rench.csv'
        csv_from_bag = location + file + topic
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

    except FileNotFoundError:
        # If csvs not available, create them fro bagfile
        # In this case, it has to read the bagfiles and extract the csvs

        b = bagreader(location + bagfile)

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

    finish_reading_topics = time.time()
    # print('Finished at: ', finish_reading_topics)
    # print('Reading files Elapsed time:', (finish_reading_topics - start_reading_topics))

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

    trial_events_states = trial_events.iloc[:, 1]

    # CALCULATE NET VALUES
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

    # Please Review these indexes for the Real Apple Picks from 0 to ___
    start_grasp_list = np.where(trial_events.iloc[:, 1] == 2)       # The ROS program publishes 2 during the GRASP
    start_grasp = start_grasp_list[0][0]                            # Just grab the first element

    start_pull_list = np.where(trial_events.iloc[:, 1] == 3)        # The ROS program publishes 3 during the PICK
    start_pull = start_pull_list[0][0]  # Just grab the first element

    finish_pull_list = np.where(trial_events.iloc[:, 1] == 4)       # The ROS program publishes 4 after the PICK
    finish_pull = finish_pull_list[0][0]  # Just grab the first element

    event_indexes = [start_grasp, start_pull, finish_pull]
    # print('The events indexes are: ', event_indexes)

    a, b, c, d, e, f, g, h = event_times(trial_events_elapsed_time,
                                         event_indexes,
                                         [f1_state_elapsed_time, f1_state_speed],
                                         [f2_state_elapsed_time, f2_state_speed],
                                         [f3_state_elapsed_time, f3_state_speed],
                                         [arm_joints_elapsed_time, joint_0_spd, joint_1_spd, joint_2_spd, joint_3_spd,
                                          joint_4_spd, joint_5_spd])

    # Define the x_min and x_max for the plots
    x_min = min(arm_time_stamp)
    x_max = max(arm_time_stamp)

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
                  'joint_0_pos', 'joint_1_pos', 'joint_2_pos', 'joint_3_pos', 'joint_4_pos', 'joint_5_pos']

    hand_labels = ['f1_acc_x', 'f1_acc_y', 'f1_acc_z', 'net_f1_acc',
                   'f2_acc_x', 'f2_acc_y', 'f2_acc_z', 'net_f2_acc',
                   'f3_acc_x', 'f3_acc_y', 'f3_acc_z', 'net_f3_acc',
                   'f1_gyro_x', 'f1_gyro_y', 'f1_gyro_z',
                   'f2_gyro_x', 'f2_gyro_y', 'f2_gyro_z',
                   'f3_gyro_x', 'f3_gyro_y', 'f3_gyro_z',
                   'f1_state_position', 'f1_state_speed', 'f1_state_effort',
                   'f2_state_position', 'f2_state_speed', 'f2_state_effort',
                   'f3_state_position', 'f3_state_speed', 'f3_state_effort']

    # Apply filter to all the variables
    arm_variables_filtered = filter_variables(arm_variables, arm_filter_param)
    hand_variables_filtered = filter_variables(hand_variables, hand_filter_param)

    # ------------------------------------  Step 5: Plot Results -------------------------------------------------------
    #
    # First select the kind of x-axis time that you want: (a) Original Time Stamps (b) human-readable time stamp

    # (a) Original Time Stamps
    # arm_time_ref = arm_time_stamp
    # arm_joints_time_ref = arm_joints_time_stamp
    # f1_state_time_ref = f1_state_time_stamp
    # f1_imu_time_ref = f1_imu_time_stamp
    # f2_state_time_ref = f2_state_time_stamp
    # f2_imu_time_ref = f2_imu_time_stamp
    # f3_state_time_ref = f3_state_time_stamp
    # f3_imu_time_ref = f3_imu_time_stamp
    # trial_events_time_ref = trial_events_time_stamp

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

    title = file + ' (Median Filter - Arm = ' + "{:.0f}".format(arm_filter_param) + \
            ', Hand = ' + str(hand_filter_param) + ') ' + pdf_variable

    # ARM figures

    # Arm's Forces
    if pdf_variable == '1 - UR5e - (FT) Wrist Forces':
        broken_axes(plot_number, axrray,
                    arm_time_ref,
                    [arm_variables_filtered[0], arm_variables_filtered[1], arm_variables_filtered[2], arm_variables_filtered[3]],
                    [arm_labels[0], arm_labels[1], arm_labels[2], arm_labels[3]],
                    e, f, g, h, title, 'Force [N]', file_label, -20, 40)

    # Arm's Torques
    elif pdf_variable == '2 - UR5e - (FT) Wrist Torques':
        broken_axes(plot_number, axrray,
                    arm_time_ref,
                    [arm_variables_filtered[4], arm_variables_filtered[5], arm_variables_filtered[6], arm_variables_filtered[7]],
                    [arm_labels[4], arm_labels[5], arm_labels[6], arm_labels[7]],
                    e, f, g, h, title, 'Torque [Nm]', file_label, -2, 2)

    # Arm's Joints
    elif pdf_variable == '3 - UR5e - Joint Angular Positions':
        broken_axes(plot_number, axrray,
                    arm_joints_time_ref,
                    [arm_variables_filtered[8], arm_variables_filtered[9], arm_variables_filtered[10], arm_variables_filtered[11], arm_variables_filtered[12], arm_variables_filtered[13]],
                    [arm_labels[8], arm_labels[9], arm_labels[10], arm_labels[11], arm_labels[12], arm_labels[13]],
                    e, f, g, h, title, 'Angular Positions [rad]', file_label, -6, 6)

    # Finger 1's Joint States
    elif pdf_variable == '4 - Finger 1 - Joint States':
        broken_axes(plot_number, axrray, f1_state_time_ref,
                    [hand_variables_filtered[21], hand_variables_filtered[22], hand_variables_filtered[23]],
                    [hand_labels[21], hand_labels[22], hand_labels[23]],
                    e, f, g, h, title, 'f1 - States', file_label, -500, 500)

    # Finger 1's accelerometers
    elif pdf_variable == '5 - Finger 1 - (IMU) Linear Acceleration':
        broken_axes(plot_number, axrray,
                    f1_imu_time_ref,
                    [hand_variables_filtered[0], hand_variables_filtered[1], hand_variables_filtered[2], hand_variables_filtered[3]],
                    [hand_labels[0], hand_labels[1], hand_labels[2], hand_labels[3]],
                    e, f, g, h, title, 'Linear Acceleration [g]', file_label, -30, 30)

    # Finger 1's Gyroscopes
    elif pdf_variable == '6 - Finger 1 - (IMU) Angular Velocity':
        broken_axes(plot_number, axrray,
                    f1_imu_time_ref,
                    [hand_variables_filtered[12], hand_variables_filtered[13], hand_variables_filtered[14]],
                    [hand_labels[12], hand_labels[13], hand_labels[14]],
                    e, f, g, h, title, 'Angular Velocity [deg/s]', file_label, -300, 300)

    # Finger 2's Joint States
    elif pdf_variable == '7 - Finger 2 - Joint States':
        broken_axes(plot_number, axrray,
                    f2_state_time_ref,
                    [hand_variables_filtered[24], hand_variables_filtered[25], hand_variables_filtered[26]],
                    [hand_labels[24], hand_labels[25], hand_labels[26]],
                    e, f, g, h, title, 'f2 - States', file_label, -500, 500)

    # Finger 2's Accelerometers
    elif pdf_variable == '8 - Finger 2 - (IMU) Linear Acceleration':
        broken_axes(plot_number, axrray,
                    f2_imu_time_ref,
                    [hand_variables_filtered[4], hand_variables_filtered[5], hand_variables_filtered[6], hand_variables_filtered[7]],
                    [hand_labels[4], hand_labels[5], hand_labels[6], hand_labels[7]],
                    e, f, g, h, title, 'Linear Acceleration [g]', file_label, -30, 30)

    # Finger 2's Gyroscopes
    elif pdf_variable == '9 - Finger 2 - (IMU) Angular Velocity':
        broken_axes(plot_number, axrray,
                    f2_imu_time_ref,
                    [hand_variables_filtered[15], hand_variables_filtered[16], hand_variables_filtered[17]],
                    [hand_labels[15], hand_labels[16], hand_labels[17]],
                    e, f, g, h, title, 'Angular Velocity [deg/s]', file_label, -300, 300)

    # Finger 3's Joint States
    elif pdf_variable == '10 - Finger 3 - Joint States':
        broken_axes(plot_number, axrray,
                    f3_state_time_ref,
                    [hand_variables_filtered[27], hand_variables_filtered[28], hand_variables_filtered[29]],
                    [hand_labels[27], hand_labels[28], hand_labels[29]],
                    e, f, g, h, title, 'f3 - States', file_label, -500, 500)

    # Finger 3's Accelerometers
    elif pdf_variable == '11 - Finger 3 - (IMU) Linear Acceleration':
        broken_axes(plot_number, axrray,
                    f3_imu_time_ref,
                    [hand_variables_filtered[8], hand_variables_filtered[9], hand_variables_filtered[10], hand_variables_filtered[11]],
                    [hand_labels[8], hand_labels[9], hand_labels[10], hand_labels[11]],
                    e, f, g, h, title, 'Linear Acceleration [g]', file_label, -30, 30)

    # Finger 3's Gyroscopes
    elif pdf_variable == '12 - Finger 3 - (IMU) Angular Velocity':
        broken_axes(plot_number, axrray,
                    f3_imu_time_ref,
                    [hand_variables_filtered[18], hand_variables_filtered[19], hand_variables_filtered[20]],
                    [hand_labels[18], hand_labels[19], hand_labels[20]],
                    e, f, g, h, title, 'Angular Velocity [deg/s]', file_label, -300, 300)

    # Experiment Events (e.g. Grasp, Pick)
    elif pdf_variable == '13 - Events during the experiment':

        states = []
        for i in range(len(trial_events_states)):
            states.append(trial_events_states[i])

        # print(trial_events_time_ref, '\n', states)

        broken_axes(plot_number, axrray,
                    trial_events_time_ref,
                    [states],
                    'events',
                    e, f, g, h, title, 'Event', file_label, -1, 8)

    # plt.show()


def look_at_labels(location, filename):
    """
    Looks for the csv file of the bagfile, in order to read the labels from the metadata
    :param location:
    :param prefix: the name of the file
    :return:
    """

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


if __name__ == '__main__':

    # --- List the variables for which you want to generate the pdfs
    variables = [
                 '1 - UR5e - (FT) Wrist Forces',
                 '2 - UR5e - (FT) Wrist Torques',
                 '3 - UR5e - Joint Angular Positions',
                 '4 - Finger 1 - Joint States',
                 '5 - Finger 1 - (IMU) Linear Acceleration',
                 '6 - Finger 1 - (IMU) Angular Velocity',
                 '7 - Finger 2 - Joint States',
                 '8 - Finger 2 - (IMU) Linear Acceleration',
                 '9 - Finger 2 - (IMU) Angular Velocity',
                 '10 - Finger 3 - Joint States',
                 '11 - Finger 3 - (IMU) Linear Acceleration',
                 '12 - Finger 3 - (IMU) Angular Velocity',
                 '13 - Events during the experiment'
                 ]

    # --- Signal Filter parameters
    # Median Filter
    hand_filter_window = 3  # Have in mind that the Sampling Rate of the Hand was 70Hz
    arm_filter_window = int(hand_filter_window * 500 / 70)  # Have in mind that the Sampling Rate of the Arm was 500Hz

    # --- Target label
    # Note: It has to be same as in function def look_at_labels
    labels = ['(Successful-Pick)', '(Failed-Pick)']
    # labels = ['(Successful-Pick)']
    # labels = ['(Failed-Pick)']

    for label in labels:

        for pdf_variable in variables:

            # --- Step 1: Create PDF
            pdf_pages = PdfPages(str(pdf_variable) + ' ' + label + ' (Median Filter - Arm = ' +
                                 "{:.0f}".format(arm_filter_window) + ', Hand = ' + str(hand_filter_window) + ').pdf')

            # --- Step 2: Create the plot array
            nb_plots = 260
            nb_plots_per_page = 4
            f, axrray = plt.subplots(nb_plots_per_page, 2, figsize=(8.5, 14), dpi=100, sharey=True)
            plt.subplots_adjust(left=0.075, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.175)
            plot_number = 1

            # --- Step 3: Generate all the plots for that particular variable within all the picks
            n_picks = 260

            # Data valid for Grasp and Pick
            # location = '/home/avl/PycharmProjects/AppleProxy/1_data_valid_for_grasp_and_pick/'

            # Data valid for Grasp
            # location = '/home/avl/PycharmProjects/AppleProxy/2_data_valid_for_grasp/'

            # Data valid for Pick
            # location = '/home/avl/PycharmProjects/AppleProxy/3_data_valid_for_pick/'

            # Not useful
            location = '/home/avl/PycharmProjects/AppleProxy/0_data_not useful/'

            print('\nPlotting ', pdf_variable)
            for i in tqdm(range(77)):
                for j in range(13):
                    file = 'apple_proxy_pick' + str(i) + '-' + str(j) + '_metadata.csv'
                    # print(file)
                    if exists(location + file):

                        file_label = look_at_labels(location, file)
                        # print("\n" + file_label)
                        # print(file)

                        if file_label == label:

                            try:

                                # Look for the corresponding bag file
                                bagfile = str(file)
                                bag_file = bagfile.replace('_metadata.csv', '.bag', 1)
                                # print(bag_file)

                                generate_plots(location, bag_file, pdf_variable, plot_number, axrray, arm_filter_window, hand_filter_window, )
                                plot_number += 1

                                # --- Create a new page if the number of rows has been reached
                                if (plot_number - 1) % nb_plots_per_page == 0 or (plot_number - 1) == nb_plots:
                                    # print('Page saved\n')
                                    pdf_pages.savefig(f)
                                    f, axrray = plt.subplots(nb_plots_per_page, 2, figsize=(8.5, 14), dpi=100, sharey=True)
                                    # Margins
                                    plt.subplots_adjust(left=0.075, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.175)


                            except KeyboardInterrupt:
                                # --- You may cancel the process at any time, and the plots will still be saved
                                pdf_pages.savefig(f)
                                print('User cancelled the process. PDF saved')
                                pdf_pages.close()

            # print('Page saved\n')
            pdf_pages.savefig(f)

            # print('PDF saved')
            pdf_pages.close()