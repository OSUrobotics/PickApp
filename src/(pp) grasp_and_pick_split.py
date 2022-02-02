"""
Takes data from the apple pick experiments and splits it in two halves: Grasp and Pick
This is useful to:
 - Get rid of the points sampled during the rest of the time (e.g. amidst labeling)
 - Have the data split for grasp or pick analyses independently
"""
# @Time : 1/26/2022 11:09 AM
# @Author : Alejandro Velasquez


import os
import pandas as pd
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter, median_filter


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


def event_times(events_elapsed_time, event_indexes):
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

    # Grasp Events
    start_grasp_apple_event_index = event_indexes[0]
    g_on = events_elapsed_time[start_grasp_apple_event_index]
    g_off = g_on + 2.0

    # Pick Events
    start_pulling_apple_event_index = event_indexes[1]
    finish_pulling_apple_event_index = event_indexes[2]
    p_on = events_elapsed_time[start_pulling_apple_event_index]
    p_off = events_elapsed_time[finish_pulling_apple_event_index]

    return g_on, g_off, p_on, p_off


def grasp_pick_indexes(time_list, grasp_start_time, grasp_end_time, pick_start_time, pick_end_time):
    """
    Given the time list of certain variable, this functions finds the indexes from that list with the times that
    match the start and end times of grasp and pick
    :param time_list:
    :param grasp_start_time:
    :param grasp_end_time:
    :param pick_start_time:
    :param pick_end_time:
    :return:
    """

    offset = 0

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


# Location in Box Folder (Alejo's laptop)
location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/Apple Proxy Picks/Winter 2022/1_data_valid_for_grasp_and_pick/'
# location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/Apple Proxy Picks/Winter 2022/2_data_valid_for_grasp/'

# stage = 'grasp'
stage = 'pick'

for subfolder in sorted(os.listdir(location)):

    # Check that the object is a sub-folder
    if os.path.isdir(location + subfolder):

        print(subfolder)

        # a. Arm's Wrench topic: forces and torques
        topic = '/rench.csv'
        csv_from_bag = location + subfolder + topic
        arm_wrench = pd.read_csv(csv_from_bag)

        # b. Arm's Joint_States topic: angular positions of the 6 joints
        topic = '/oint_states.csv'
        csv_from_bag = location + subfolder + topic
        arm_joints = pd.read_csv(csv_from_bag)

        # c. Hand's finger1 imu
        topic = '/applehand-finger1-imu.csv'
        csv_from_bag = location + subfolder + topic
        f1_imu = pd.read_csv(csv_from_bag)

        # d. Hand's finger1 joint state
        topic = '/applehand-finger1-jointstate.csv'
        csv_from_bag = location + subfolder + topic
        f1_state = pd.read_csv(csv_from_bag)

        # e. Hand's finger2 imu
        topic = '/applehand-finger2-imu.csv'
        csv_from_bag = location + subfolder + topic
        f2_imu = pd.read_csv(csv_from_bag)

        # f. Hand's finger2 joint state
        topic = '/applehand-finger2-jointstate.csv'
        csv_from_bag = location + subfolder + topic
        f2_state = pd.read_csv(csv_from_bag)

        # g. Hand's finger3 imu
        topic = '/applehand-finger3-imu.csv'
        csv_from_bag = location + subfolder + topic
        f3_imu = pd.read_csv(csv_from_bag)

        # h. Hand's finger3 joint state
        topic = '/applehand-finger3-jointstate.csv'
        csv_from_bag = location + subfolder + topic
        f3_state = pd.read_csv(csv_from_bag)

        # EVENT's topic
        # i. Trial events
        topic = '/apple_trial_events.csv'
        csv_from_bag = location + subfolder + topic
        events = pd.read_csv(csv_from_bag)

        # ----------------------  Step 2: Extract each vector from the csv's, and adjust the time-----------------------
        # TIME STAMPS
        arm_time_stamp = arm_wrench.iloc[:, 0]
        arm_joints_time_stamp = arm_joints.iloc[:, 0]
        f1_imu_time_stamp = f1_imu.iloc[:, 0]
        f1_state_time_stamp = f1_state.iloc[:, 0]
        f2_imu_time_stamp = f2_imu.iloc[:, 0]
        f2_state_time_stamp = f2_state.iloc[:, 0]
        f3_imu_time_stamp = f3_imu.iloc[:, 0]
        f3_state_time_stamp = f3_state.iloc[:, 0]
        events_time_stamp = events.iloc[:, 0]

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
        # net_force = net_value(forces_x, forces_y, forces_z)
        # net_torque = net_value(torques_x, torques_y, torques_z)
        # net_f1_acc = net_value(f1_acc_x, f1_acc_y, f1_acc_z)
        # net_f2_acc = net_value(f2_acc_x, f2_acc_y, f2_acc_z)
        # net_f3_acc = net_value(f3_acc_x, f3_acc_y, f3_acc_z)

        arm_elapsed_time = elapsed_time(forces_x, arm_time_stamp)
        arm_joints_elapsed_time = elapsed_time(arm_joints, arm_joints_time_stamp)
        f1_imu_elapsed_time = elapsed_time(f1_imu, f1_imu_time_stamp)
        f1_state_elapsed_time = elapsed_time(f1_state, f1_state_time_stamp)
        f2_imu_elapsed_time = elapsed_time(f2_imu, f2_imu_time_stamp)
        f2_state_elapsed_time = elapsed_time(f2_state, f2_state_time_stamp)
        f3_imu_elapsed_time = elapsed_time(f3_imu, f3_imu_time_stamp)
        f3_state_elapsed_time = elapsed_time(f3_state, f3_state_time_stamp)
        events_elapsed_time = elapsed_time(events, events_time_stamp)

        # Get the event indexes where the grasp and pick happen
        start_grasp_list = np.where(events.iloc[:, 1] == 2)  # The ROS program publishes 2 during the GRASP
        start_grasp = start_grasp_list[0][0]  # Just grab the first element

        start_pull_list = np.where(events.iloc[:, 1] == 3)  # The ROS program publishes 3 during the PICK
        start_pull = start_pull_list[0][0]  # Just grab the first element

        finish_pull_list = np.where(events.iloc[:, 1] == 4)  # The ROS program publishes 4 after the PICK
        finish_pull = finish_pull_list[0][0]  # Just grab the first element

        event_indexes = [start_grasp, start_pull, finish_pull]

        # Now that you have the event indexes, get the times at which it happened
        grasp_on, grasp_off, pick_on, pick_off = event_times(events_elapsed_time, event_indexes)

        print('The grasp started at %.3f and ended at %.3f' % (grasp_on, grasp_off))
        print('The pick  started at %.3f and ended at %.3f' % (pick_on, pick_off))

        # --------------------------------------  Step 4: Filter Data  -------------------------------------------------
        # Smooth data

        # Unify variables in a single list
        arm_variables = [forces_x, forces_y, forces_z,              # 0, 1, 2
                         torques_x, torques_y, torques_z,           # 3, 4, 5
                         joint_0_pos, joint_1_pos, joint_2_pos,     # 6, 7, 8
                         joint_3_pos, joint_4_pos, joint_5_pos]     # 9, 10, 11

        hand_variables = [f1_acc_x, f1_acc_y, f1_acc_z,     # 0, 1, 2
                          f2_acc_x, f2_acc_y, f2_acc_z,     # 3, 4, 5
                          f3_acc_x, f3_acc_y, f3_acc_z,     # 6, 7, 8
                          f1_gyro_x, f1_gyro_y, f1_gyro_z,  # 9, 10, 11
                          f2_gyro_x, f2_gyro_y, f2_gyro_z,  # 12, 13, 14,
                          f3_gyro_x, f3_gyro_y, f3_gyro_z,  # 15, 16, 17,
                          f1_state_position, f1_state_speed, f1_state_effort,  # 18, 19, 20
                          f2_state_position, f2_state_speed, f2_state_effort,  # 21, 22, 23,
                          f3_state_position, f3_state_speed, f3_state_effort]  # 24, 25, 26,

        # Apply filter to all the variables
        # Median Filter
        hand_filter_window = 3  # Have in mind that the Sampling Rate of the Hand was 70Hz
        arm_filter_window = int(hand_filter_window * 500 / 70)  # Have in mind that the Sampling Rate of the Arm was 500Hz

        arm_variables_filtered = filter_variables(arm_variables, arm_filter_window)
        hand_variables_filtered = filter_variables(hand_variables, hand_filter_window)

        # ------------------------------------ Step 5: Crop Data for csv's ----------------------------------------
        # (b) Human Readable Time Stamps
        arm_time_ref = arm_elapsed_time
        arm_joints_time_ref = arm_joints_elapsed_time
        f1_state_time_ref = f1_state_elapsed_time
        f1_imu_time_ref = f1_imu_elapsed_time
        f2_state_time_ref = f2_state_elapsed_time
        f2_imu_time_ref = f2_imu_elapsed_time
        f3_state_time_ref = f3_state_elapsed_time
        f3_imu_time_ref = f3_imu_elapsed_time
        trial_events_time_ref = events_elapsed_time

        # --- ARM WRENCH ---
        # Get the indexes of the grasp and pick moments, in order to obtain the slices
        g_on, g_off, p_on, p_off = grasp_pick_indexes(arm_time_ref, grasp_on, grasp_off, pick_on, pick_off)
        header = 'elapsed time, force_x, force_y, force_z, torque_x, torque_y, torque_z'

        # Choose the time window whether for GRASP or PICK
        if stage == 'grasp':
            t_on = g_on
            t_off = g_off
        else:
            t_on = p_on
            t_off = p_off

        # Grasp
        arm_list = [arm_time_ref[t_on:t_off],
                    arm_variables_filtered[0][t_on:t_off],  # Slice the list only during the grasp
                    arm_variables_filtered[1][t_on:t_off],  # Slice the list only during the grasp
                    arm_variables_filtered[2][t_on:t_off],  # Slice the list only during the grasp
                    arm_variables_filtered[3][t_on:t_off],
                    arm_variables_filtered[4][t_on:t_off],
                    arm_variables_filtered[5][t_on:t_off]
                    ]
        np.savetxt("csvs/" + str(subfolder) + "_" + stage + "_wrench.csv", np.array(arm_list).T, delimiter=',',
                   header=header)

        # --- HAND FINGER 1 ---
        # Get the indexes of the grasp and pick moments, in order to obtain the slices
        g_on, g_off, p_on, p_off = grasp_pick_indexes(f1_state_time_ref, grasp_on, grasp_off, pick_on, pick_off)
        header = 'elapsed_time, f1_state_position, f1_state_speed, f1_state_effort'

        # Choose the time window whether for GRASP or PICK
        if stage == 'grasp':
            t_on = g_on
            t_off = g_off
        else:
            t_on = p_on
            t_off = p_off

        # Grasp
        f1_states_list = [f1_state_time_ref[t_on:t_off],
                          hand_variables_filtered[18][t_on:t_off],
                          hand_variables_filtered[19][t_on:t_off],
                          hand_variables_filtered[20][t_on:t_off]
                          ]
        np.savetxt("csvs/" + str(subfolder) + "_" + stage + "_f1_states.csv", np.array(f1_states_list).T, delimiter=',',
                   header=header)

        # Get the indexes of the grasp and pick moments, in order to obtain the slices
        g_on, g_off, p_on, p_off = grasp_pick_indexes(f1_imu_time_ref, grasp_on, grasp_off, pick_on, pick_off)
        header = 'elapsed time, f1_acc_x, f1_acc_y, f1_acc_z, f1_gyro_x, f1_gyro_y, f1_gyro_z'

        # Choose the time window whether for GRASP or PICK
        if stage == 'grasp':
            t_on = g_on
            t_off = g_off
        else:
            t_on = p_on
            t_off = p_off

        # Grasp
        f1_imu_list = [f1_imu_time_ref[t_on:t_off],
                       hand_variables_filtered[0][t_on:t_off],
                       hand_variables_filtered[1][t_on:t_off],
                       hand_variables_filtered[2][t_on:t_off],
                       hand_variables_filtered[9][t_on:t_off],
                       hand_variables_filtered[10][t_on:t_off],
                       hand_variables_filtered[11][t_on:t_off],
                       ]
        np.savetxt("csvs/" + str(subfolder) + "_" + stage + "_f1_imu.csv", np.array(f1_imu_list).T, delimiter=',',
                   header=header)


        # --- HAND FINGER 2 ---
        # Get the indexes of the grasp and pick moments, in order to obtain the slices
        g_on, g_off, p_on, p_off = grasp_pick_indexes(f2_state_time_ref, grasp_on, grasp_off, pick_on, pick_off)
        header = 'elapsed_time, f2_state_position, f2_state_speed, f2_state_effort'

        # Choose the time window whether for GRASP or PICK
        if stage == 'grasp':
            t_on = g_on
            t_off = g_off
        else:
            t_on = p_on
            t_off = p_off

        # Grasp
        f2_states_list = [f2_state_time_ref[t_on:t_off],
                          hand_variables_filtered[21][t_on:t_off],
                          hand_variables_filtered[22][t_on:t_off],
                          hand_variables_filtered[23][t_on:t_off]
                          ]
        np.savetxt("csvs/" + str(subfolder) + "_" + stage + "_f2_states.csv", np.array(f2_states_list).T, delimiter=',',
                   header=header)

        # Get the indexes of the grasp and pick moments, in order to obtain the slices
        g_on, g_off, p_on, p_off = grasp_pick_indexes(f2_imu_time_ref, grasp_on, grasp_off, pick_on, pick_off)
        header = 'elapsed time, f2_acc_x, f2_acc_y, f2_acc_z, f2_gyro_x, f2_gyro_y, f2_gyro_z'

        # Choose the time window whether for GRASP or PICK
        if stage == 'grasp':
            t_on = g_on
            t_off = g_off
        else:
            t_on = p_on
            t_off = p_off

        # Grasp
        f2_imu_list = [f2_imu_time_ref[t_on:t_off],
                       hand_variables_filtered[3][t_on:t_off],
                       hand_variables_filtered[4][t_on:t_off],
                       hand_variables_filtered[5][t_on:t_off],
                       hand_variables_filtered[12][t_on:t_off],
                       hand_variables_filtered[13][t_on:t_off],
                       hand_variables_filtered[14][t_on:t_off]
                       ]
        np.savetxt("csvs/" + str(subfolder) + "_" + stage + "_f2_imu.csv", np.array(f2_imu_list).T, delimiter=',',
                   header=header)

        # --- HAND FINGER 3 ---
        # Get the indexes of the grasp and pick moments, in order to obtain the slices
        g_on, g_off, p_on, p_off = grasp_pick_indexes(f3_state_time_ref, grasp_on, grasp_off, pick_on, pick_off)
        header = 'elapsed_time, f3_state_position, f3_state_speed, f3_state_effort'

        # Choose the time window whether for GRASP or PICK
        if stage == 'grasp':
            t_on = g_on
            t_off = g_off
        else:
            t_on = p_on
            t_off = p_off

        # Grasp
        f3_states_list = [f3_state_time_ref[t_on:t_off],
                          hand_variables_filtered[24][t_on:t_off],
                          hand_variables_filtered[25][t_on:t_off],
                          hand_variables_filtered[26][t_on:t_off]
                          ]
        np.savetxt("csvs/" + str(subfolder) + "_" + stage + "_f3_states.csv", np.array(f3_states_list).T, delimiter=',',
                   header=header)
        # Grasp
        # Get the indexes of the grasp and pick moments, in order to obtain the slices
        g_on, g_off, p_on, p_off = grasp_pick_indexes(f3_imu_time_ref, grasp_on, grasp_off, pick_on, pick_off)
        header = 'elapsed time, f3_acc_x, f3_acc_y, f3_acc_z, f3_gyro_x, f3_gyro_y, f3_gyro_z'

        # Choose the time window whether for GRASP or PICK
        if stage == 'grasp':
            t_on = g_on
            t_off = g_off
        else:
            t_on = p_on
            t_off = p_off

        # Grasp
        f3_imu_list = [f3_imu_time_ref[t_on:t_off],
                       hand_variables_filtered[6][t_on:t_off],
                       hand_variables_filtered[7][t_on:t_off],
                       hand_variables_filtered[8][t_on:t_off],
                       hand_variables_filtered[15][t_on:t_off],
                       hand_variables_filtered[16][t_on:t_off],
                       hand_variables_filtered[17][t_on:t_off],
                       ]
        np.savetxt("csvs/" + str(subfolder) + "_" + stage + "_f3_imu.csv", np.array(f3_imu_list).T, delimiter=',',
                   header=header)



