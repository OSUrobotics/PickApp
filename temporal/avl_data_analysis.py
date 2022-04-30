# ... System related packages
import inspect
import os
import time
# ... File related packages
import pandas as pd
import csv
import bagpy
from bagpy import bagreader
# ... Math related packages
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
from scipy.stats import skew
import statistics as st
from numpy import trapz
# ... Plot related packages
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# sns.set()  # Setting seaborn as default style even if use only matplotlib
plt.close('all')


class Apples:

    def __init__(self):
        # ----------------------------- Step 1: Define the location of the data -- -------------------------------------
        # Location
        # location = '/media/avl/StudyData/ApplePicking Data/5 - Real Apple with Hand Closing Fixed/bagfiles'  # External SSD
        # location = '/home/avl/PycharmProjects/icra22/bagfiles/'       # Lab's laptop
        # location = '/home/avl/ur_ws/src/apple_proxy/bag_files'        # Lab's PC for ROS related
        # location = '/home/avl/PycharmProjects/AppleProxy/bagfiles/'     # Lab's PC
        self.location = '/home/avl/PycharmProjects/appleProxy/bagfiles/'  # Alejo's laptop

        # Topics to read
        self.arm_wrench = []
        self.arm_joints = []
        self.f1_imu = []
        self.f1_state = []
        self.f2_imu = []
        self.f2_state = []
        self.f3_imu = []
        self.f3_state = []
        self.trial_events = []

        # Lists of Variables for Arm and Hand
        self.arm_variables = []
        self.arm_variables_filtered = []
        self.arm_variables_grasp = []
        self.arm_variables_grasp_filtered = []
        self.arm_variables_pick = []
        self.arm_variables_pick_filtered = []

        self.hand_variables = []
        self.hand_variables_filtered = []
        self.hand_variables_grasp = []
        self.hand_variables_grasp_filtered = []
        self.hand_variables_pick = []
        self.hand_variables_pick_filtered = []

        self.arm_labels = ['forces_x', 'forces_y', 'forces_z', 'net_force',
                          'torques_x', 'torques_y', 'torques_z', 'net_torque',
                          'joint_0_pos', 'joint_1_pos', 'joint_2_pos', 'joint_3_pos', 'joint_4_pos',
                          'joint_5_pos']  # 8, 9, 10, 11, 12, 13

        self.hand_labels = ['f1_acc_x', 'f1_acc_y', 'f1_acc_z', 'net_f1_acc',  # 0, 1, 2, 3
                           'f2_acc_x', 'f2_acc_y', 'f2_acc_z', 'net_f2_acc',  # 4, 5, 6, 7
                           'f3_acc_x', 'f3_acc_y', 'f3_acc_z', 'net_f3_acc',  # 8, 9, 10, 11
                           'f1_gyro_x', 'f1_gyro_y', 'f1_gyro_z',
                           'f2_gyro_x', 'f2_gyro_y', 'f2_gyro_z',
                           'f3_gyro_x', 'f3_gyro_y', 'f3_gyro_z',
                           'f1_state_position', 'f1_state_speed', 'f1_state_effort',
                           'f2_state_position', 'f2_state_speed', 'f2_state_effort',
                           'f3_state_position', 'f3_state_speed', 'f3_state_effort']

        self.arm_elapsed_time = 0
        self.arm_joints_elapsed_time = 0
        self.f1_imu_elapsed_time = 0
        self.f1_state_elapsed_time = 0
        self.f2_imu_elapsed_time = 0
        self.f2_state_elapsed_time = 0
        self.f3_imu_elapsed_time = 0
        self.f3_state_elapsed_time = 0
        self.trial_events_elapsed_time = 0

        # Event times
        self.grasp_start = 0
        self.grasp_end = 0
        self.pick_start = 0
        self.pick_end = 0

        # Store the features to analyze
        # 1. SUCCESS
        # 1.1.  ARM-GRASP
        # 1.1.1.    FORCE
        self.max_success_grasp = []
        self.mean_success_grasp = []
        self.median_success_grasp = []
        self.force_skew_success_grasp = []
        self.fz_max_success_grasp = []
        self.netforce_auc_success_grasp = []

        # 1.2.  ARM-PICK
        # 1.2.1.    FORCE
        self.max_success_pick = []
        self.mean_success_pick = []
        self.median_success_pick = []
        self.force_skew_success_pick = []
        self.fz_max_success_pick = []
        self.netforce_auc_success_pick = []

        # 1.3.  HAND-GRASP
        self.f1_max_success_grasp = []
        self.f1_gyro_auc_success_grasp = []  # AUC: Area Under Curve

        # 1.4.  HAND-PICK
        self.f1_max_success_pick = []
        self.f1_gyro_auc_success_pick = []  # AUC: Area Under Curve


        # 2. FAILURE
        # 2.1. ARM-GRASP
        self.max_failed_grasp = []
        self.mean_failed_grasp = []
        self.median_failed_grasp = []
        self.force_skew_failed_grasp = []
        self.fz_max_failed_grasp = []
        self.netforce_auc_failed_grasp = []

        # 2.2. ARM-PICK
        self.max_failed_pick = []
        self.mean_failed_pick = []
        self.median_failed_pick = []
        self.force_skew_failed_pick = []
        self.fz_max_failed_pick = []
        self.netforce_auc_failed_pick = []

        # 2.3. HAND-GRASP
        self.f1_max_failed_grasp = []
        self.f1_gyro_auc_failed_grasp = []  # AUC: Area Under Curve

        # 2.4. HAND-PICK
        self.f1_max_failed_pick = []
        self.f1_gyro_auc_failed_pick = []  # AUC: Area Under Curve


        self.crop_tolerance = 0.5

    def crop_data(self, data_array, time_array):

        # print('Data Lengths are:', len(data_array), len(time_array))

        crop_grasp = []
        crop_grasp_time = []
        crop_pick = []
        crop_pick_time = []
        for i in range(len(time_array)):
            # print(i)
            # Store the Pick values
            if (self.grasp_start - self.crop_tolerance) < time_array[i] < (self.grasp_end + self.crop_tolerance):
                crop_grasp.append(data_array[i])
                crop_grasp_time.append(time_array[i])
            # Store the Grasp values
            elif (self.pick_start - self.crop_tolerance) < time_array[i] < (self.pick_end + self.crop_tolerance):
                crop_pick.append(data_array[i])
                crop_pick_time.append(time_array[i])

        return crop_grasp, crop_grasp_time, crop_pick, crop_pick_time

    def elapsed_time(self, variable, time_stamp):
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

    def event_times(self, trial_events_elapsed_time, event_indexes, f1, f2, f3, arm):
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
        i1, e1 = self.find_instance(f1_state_speed, f1_state_elapsed_time, 0.01, closing_hand_event_time, 'starts')
        i2, e2 = self.find_instance(f2_state_speed, f2_state_elapsed_time, 0.01, closing_hand_event_time, 'starts')
        i3, e3 = self.find_instance(f3_state_speed, f3_state_elapsed_time, 0.01, closing_hand_event_time, 'starts')
        e = min(e1, e2, e3)
        # print('\nFinger servos start moving at: %.2f, %.2f and %.2f ' % (e1, e2, e3))
        # print('The time delay between event and servo moving is: %.2f' % (e - b))

        # Servos Stop Moving Event
        # Find the instance when the finger's motors stop indeed moving
        j1, f1 = self.find_instance(f1_state_speed, f1_state_elapsed_time, 0.01, e, 'stops')
        j2, f2 = self.find_instance(f2_state_speed, f2_state_elapsed_time, 0.01, e, 'stops')
        j3, f3 = self.find_instance(f3_state_speed, f3_state_elapsed_time, 0.01, e, 'stops')
        f = max(f1, f2, f3)
        # print('Finger servos stop moving at: %.2f, %.2f and %.2f' % (f1, f2, f3))

        if len(event_indexes) == 4:
            c = f

        k0, g0 = self.find_instance(joint_0_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
        k1, g1 = self.find_instance(joint_1_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
        k2, g2 = self.find_instance(joint_2_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
        k3, g3 = self.find_instance(joint_3_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
        k4, g4 = self.find_instance(joint_4_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
        k5, g5 = self.find_instance(joint_5_spd, arm_joints_elapsed_time, 0.01, c, 'starts')
        g = min(g0, g1, g2, g3, g4, g5)
        # print(
        #     "The times at which the UR5 joints start are: %.2f, %.2f, %2.f, %2.f, %.2f and %.2f" % (g0, g1, g2, g3, g4, g5))
        # print('\nUR5 starts moving at: %.2f ' % g)

        if len(event_indexes) == 4:
            c = g

        k = max(g0, g1, g2, g3, g4, g5)
        # print("The values of k are: %.2f, %.2f, %2.f, %2.f, %.2f and %.2f" % (k0, k1, k2, k3, k4, k5))

        # Arm Stops pulling apple
        l0, h0 = self.find_instance(joint_0_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
        l1, h1 = self.find_instance(joint_1_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
        l2, h2 = self.find_instance(joint_2_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
        l3, h3 = self.find_instance(joint_3_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
        l4, h4 = self.find_instance(joint_4_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
        l5, h5 = self.find_instance(joint_5_spd, arm_joints_elapsed_time, 0.001, g, 'stops')
        h = max(h0, h1, h2, h3, h4, h5)
        # print(
        #     "The times at which the UR5 joints stop are: %.2f, %.2f, %2.f, %2.f, %.2f and %.2f" % (h0, h1, h2, h3, h4, h5))
        # print('UR5 stops moving at: %.2f' % h)

        return e, f, g, h

    def extract_variables(self):

        # -----------------------  Step 1: Extract each vector from the csv's, and adjust the time----------------------
        # TIME STAMPS
        arm_time_stamp = self.arm_wrench.iloc[:, 0]
        arm_joints_time_stamp = self.arm_joints.iloc[:, 0]
        f1_imu_time_stamp = self.f1_imu.iloc[:, 0]
        f1_state_time_stamp = self.f1_state.iloc[:, 0]
        f2_imu_time_stamp = self.f2_imu.iloc[:, 0]
        f2_state_time_stamp = self.f2_state.iloc[:, 0]
        f3_imu_time_stamp = self.f3_imu.iloc[:, 0]
        f3_state_time_stamp = self.f3_state.iloc[:, 0]
        trial_events_time_stamp = self.trial_events.iloc[:, 0]

        # ARM
        forces_x = self.arm_wrench.iloc[:, 5]
        forces_y = self.arm_wrench.iloc[:, 6]
        forces_z = self.arm_wrench.iloc[:, 7]
        torques_x = self.arm_wrench.iloc[:, 8]
        torques_y = self.arm_wrench.iloc[:, 9]
        torques_z = self.arm_wrench.iloc[:, 10]

        joint_0_pos = self.arm_joints.iloc[:, 6]
        joint_1_pos = self.arm_joints.iloc[:, 7]
        joint_2_pos = self.arm_joints.iloc[:, 8]
        joint_3_pos = self.arm_joints.iloc[:, 9]
        joint_4_pos = self.arm_joints.iloc[:, 10]
        joint_5_pos = self.arm_joints.iloc[:, 11]

        joint_0_spd = self.arm_joints.iloc[:, 12]
        joint_1_spd = self.arm_joints.iloc[:, 13]
        joint_2_spd = self.arm_joints.iloc[:, 14]
        joint_3_spd = self.arm_joints.iloc[:, 15]
        joint_4_spd = self.arm_joints.iloc[:, 16]
        joint_5_spd = self.arm_joints.iloc[:, 17]

        # HAND
        f1_state_position = self.f1_state.iloc[:, 5]
        f1_state_speed = self.f1_state.iloc[:, 6]
        f1_state_effort = self.f1_state.iloc[:, 7]

        f2_state_position = self.f2_state.iloc[:, 5]
        f2_state_speed = self.f2_state.iloc[:, 6]
        f2_state_effort = self.f2_state.iloc[:, 7]

        f3_state_position = self.f3_state.iloc[:, 5]
        f3_state_speed = self.f3_state.iloc[:, 6]
        f3_state_effort = self.f3_state.iloc[:, 7]

        f1_acc_x = self.f1_imu.iloc[:, 5]
        f1_acc_y = self.f1_imu.iloc[:, 6]
        f1_acc_z = self.f1_imu.iloc[:, 7]

        f2_acc_x = self.f2_imu.iloc[:, 5]
        f2_acc_y = self.f2_imu.iloc[:, 6]
        f2_acc_z = self.f2_imu.iloc[:, 7]

        f3_acc_x = self.f3_imu.iloc[:, 5]
        f3_acc_y = self.f3_imu.iloc[:, 6]
        f3_acc_z = self.f3_imu.iloc[:, 7]

        f1_gyro_x = self.f1_imu.iloc[:, 8]
        f1_gyro_y = self.f1_imu.iloc[:, 9]
        f1_gyro_z = self.f1_imu.iloc[:, 10]

        f2_gyro_x = self.f2_imu.iloc[:, 8]
        f2_gyro_y = self.f2_imu.iloc[:, 9]
        f2_gyro_z = self.f2_imu.iloc[:, 10]

        f3_gyro_x = self.f3_imu.iloc[:, 8]
        f3_gyro_y = self.f3_imu.iloc[:, 9]
        f3_gyro_z = self.f3_imu.iloc[:, 10]

        # Net Values
        net_force = self.net_value(forces_x, forces_y, forces_z)
        net_torque = self.net_value(torques_x, torques_y, torques_z)
        net_f1_acc = self.net_value(f1_acc_x, f1_acc_y, f1_acc_z)
        net_f2_acc = self.net_value(f2_acc_x, f2_acc_y, f2_acc_z)
        net_f3_acc = self.net_value(f3_acc_x, f3_acc_y, f3_acc_z)

        # Elapsed Times
        self.arm_elapsed_time = self.elapsed_time(forces_x, arm_time_stamp)
        self.arm_joints_elapsed_time = self.elapsed_time(self.arm_joints, arm_joints_time_stamp)
        self.f1_imu_elapsed_time = self.elapsed_time(self.f1_imu, f1_imu_time_stamp)
        self.f1_state_elapsed_time = self.elapsed_time(self.f1_state, f1_state_time_stamp)
        self.f2_imu_elapsed_time = self.elapsed_time(self.f2_imu, f2_imu_time_stamp)
        self.f2_state_elapsed_time = self.elapsed_time(self.f2_state, f2_state_time_stamp)
        self.f3_imu_elapsed_time = self.elapsed_time(self.f3_imu, f3_imu_time_stamp)
        self.f3_state_elapsed_time = self.elapsed_time(self.f3_state, f3_state_time_stamp)
        self.trial_events_elapsed_time = self.elapsed_time(self.trial_events, trial_events_time_stamp)

        # -----------------------  Step 2: Obtain the times at which we will crop the data -----------------------------
        # TIME STAMPS
        # First get the indexes when the events happen
        event_indexes = np.where(np.diff(self.trial_events.iloc[:, 1], prepend=np.nan))[0]
        # print('The events indexes are: ', event_indexes)

        self.grasp_start, self.grasp_end, self.pick_start, self.pick_end = self.event_times(self.trial_events_elapsed_time,
                                             event_indexes,
                                             [self.f1_state_elapsed_time, f1_state_speed],
                                             [self.f2_state_elapsed_time, f2_state_speed],
                                             [self.f3_state_elapsed_time, f3_state_speed],
                                             [self.arm_joints_elapsed_time, joint_0_spd, joint_1_spd, joint_2_spd,
                                              joint_3_spd,
                                              joint_4_spd, joint_5_spd])

        # -----------------------  Step 3: Crop the data and unify in lists --------------------------------------------
        # Crop the data between the values
        # Arm's Force
        forces_net_grasp_values, forces_net_grasp_times, forces_net_pick_values, forces_net_pick_times = \
            self.crop_data(net_force, self.arm_elapsed_time)
        # Arm's Force z
        forces_z_grasp_values, forces_z_grasp_times, forces_z_pick_values, forces_z_pick_times = \
            self.crop_data(forces_z, self.arm_elapsed_time)
        # Arm's Torque
        torques_net_grasp_values, torques_net_grasp_times, torques_net_pick_values, torques_net_pick_times = \
            self.crop_data(net_torque, self.arm_elapsed_time)
        # Finger 1 Accelerometer
        net_f1_acc_grasp, net_f1_acc_grasp_times, net_f1_acc_pick, net_f1_acc_pick_times = \
            self.crop_data(net_f1_acc, self.f1_imu_elapsed_time)
        # Finger 2 Accelerometer
        net_f2_acc_grasp, net_f2_acc_grasp_times, net_f2_acc_pick, net_f2_acc_pick_times = \
            self.crop_data(net_f2_acc, self.f2_imu_elapsed_time)
        # Finger 1 Gyroscope
        f1_gyro_x_grasp, f1_gyro_x_grasp_times, f1_gyro_x_pick, f1_gyro_x_pick_time = \
            self.crop_data(f2_gyro_x, self.f2_imu_elapsed_time)


        # Plot to verify that the cropping was ok
        # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # ax1.plot(forces_net_grasp_times, forces_net_grasp_values)
        # ax2.plot(forces_net_pick_times, forces_net_pick_values)
        # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # ax1.plot(torques_net_grasp_times, torques_net_grasp_values)
        # ax2.plot(torques_net_pick_times, torques_net_pick_values)
        # plt.show()

        # -----------------------  Step 4: Gather data intos lists  ----------------------------------------------------

        # Unify all the variables in a single list
        self.arm_variables = [forces_x, forces_y, forces_z, net_force,  # 0, 1, 2, 3
                              torques_x, torques_y, torques_z, net_torque,  # 4, 5, 6, 7
                              joint_0_pos, joint_1_pos, joint_2_pos, joint_3_pos, joint_4_pos,
                              joint_5_pos]  # 8, 9, 10, 11, 12, 13

        self.hand_variables = [f1_acc_x, f1_acc_y, f1_acc_z, net_f1_acc,  # 0, 1, 2, 3,
                               f2_acc_x, f2_acc_y, f2_acc_z, net_f2_acc,  # 4, 5, 6, 7,
                               f3_acc_x, f3_acc_y, f3_acc_z, net_f3_acc,  # 8, 9, 10, 11,
                               f1_gyro_x, f1_gyro_y, f1_gyro_z,  # 12, 13, 14,
                               f2_gyro_x, f2_gyro_y, f2_gyro_z,  # 15, 16, 17,
                               f3_gyro_x, f3_gyro_y, f3_gyro_z,  # 18, 19, 20,
                               f1_state_position, f1_state_speed, f1_state_effort,  # 21, 22, 23,
                               f2_state_position, f2_state_speed, f2_state_effort,  # 24, 25, 26,
                               f3_state_position, f3_state_speed, f3_state_effort]

        # Separate the data into 4 lists: ARM-GRASP, ARM-PICK, HAND-GRASP, HAND-PICK
        # 1) ARM - GRASP: Unify the piece of time series that correspond to the Grasp
        self.arm_variables_grasp = [forces_net_grasp_values,
                                    torques_net_grasp_values,
                                    forces_z_grasp_values]

        # 2) ARM - PICK: Unify the piece of the time series that correspond the the Pick
        self.arm_variables_pick = [forces_net_pick_values,
                                   torques_net_pick_values,
                                   forces_z_pick_values]

        # 3) HAND - GRASP:
        self.hand_variables_grasp = [net_f1_acc_grasp,
                                     net_f2_acc_grasp,
                                     f1_gyro_x_grasp]

        # 4) HAND - PICK:
        self.hand_variables_pick = [net_f1_acc_pick,
                                    net_f2_acc_pick,
                                    f1_gyro_x_pick]

        # Filter them
        self.arm_variables_pick_filtered = self.filter_variables(self.arm_variables_pick, 21)
        self.arm_variables_grasp_filtered = self.filter_variables(self.arm_variables_grasp, 21)

        self.hand_variables_pick_filtered = self.filter_variables(self.hand_variables_pick, 3)
        self.hand_variables_grasp_filtered = self.filter_variables(self.hand_variables_grasp, 3)

    def filter_variables(self, variables, parameter):
        """
            This function is meant to filter a list of lists, because usually these built-in functions don't do it
            """
        # Median Filter
        variables_filtered = []
        for i in range(len(variables)):
            variable_filtered = median_filter(variables[i], parameter)
            variables_filtered.append(variable_filtered)

        # Gaussian Filter
        return variables_filtered

    def find_instance(self, array, time_array, threshold, initial_time, case='starts'):
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

    def look_at_labels(self, prefix):
        """
            Looks for the csv file of the bagfile, in order to read the labels from the metadata
            :param location:
            :param prefix: the name of the file
            :return:
            """
        # --- Step 1: Look for the file
        for filename in os.listdir(self.location):
            # print(filename)
            if filename.startswith(prefix + '_'):
                # print(filename)  # print the name of the file to make sure it is what
                break

        # print('\n **** The filename is', filename)

        # --- Open the file
        with open(self.location + filename) as f:
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

    def net_value(self, var_x, var_y, var_z):
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

    def plot_features(self, apples):
        """
        Plot the boxplots for all the features
        :param apples:
        :return:
        """

        features_to_plot = [[apples.max_success_grasp, apples.max_success_pick, apples.max_failed_grasp, apples.max_failed_pick],
                            [apples.mean_success_grasp, apples.mean_success_pick, apples.mean_failed_grasp, apples.mean_failed_pick],
                            [apples.median_success_grasp, apples.median_success_pick, apples.median_failed_grasp, apples.median_failed_pick],
                            [apples.fz_max_success_grasp, apples.fz_max_success_pick, apples.fz_max_failed_grasp, apples.fz_max_failed_pick],
                            [apples.force_skew_success_grasp, apples.force_skew_success_pick, apples.force_skew_failed_grasp, apples.force_skew_failed_pick],
                            [apples.f1_gyro_auc_success_grasp, apples.f1_gyro_auc_success_pick, apples.f1_gyro_auc_failed_grasp, apples.f1_gyro_auc_failed_pick],
                            [apples.netforce_auc_success_grasp, apples.netforce_auc_success_pick, apples.netforce_auc_failed_grasp, apples.netforce_auc_failed_pick]]

        names_futures_to_plot = [['Net Force [N]', 'Max Value'],
                                 ['Net Force [N]', 'Mean Value'],
                                 ['Net Force [N]', 'Median'],
                                 ['Force z [N]', 'Max Value'],
                                 ['Net Force [N]', 'Skewness'],
                                 ['Finger1 Angular Velocity [deg/s]', 'Area under curve'],
                                 ['Net Force [N]', 'Area under Curve']]

        for i in range(len(features_to_plot)):
            fig, ax = plt.subplots()
            ax.boxplot([features_to_plot[i][0], features_to_plot[i][2]], labels=['Success', 'Failed'])
            plt.grid()
            plt.ylabel(names_futures_to_plot[i][0] + ' ' + names_futures_to_plot[i][1])
            plt.title('Comparison of ' + names_futures_to_plot[i][0] + ' ' + names_futures_to_plot[i][1] + ' during GRASP')
            fig, ax = plt.subplots()
            ax.boxplot([features_to_plot[i][1], features_to_plot[i][3]], labels=['Success', 'Failed'])
            plt.grid()
            plt.ylabel(names_futures_to_plot[i][0] + ' ' + names_futures_to_plot[i][1])
            plt.title('Comparison of ' + names_futures_to_plot[i][0] + ' ' + names_futures_to_plot[i][1] + ' during PICK')

    def plot_pdfs(self):
        a = 1

    def read_data(self, n_picks):

        for pick in range(1, n_picks + 1):

            # Skip picks that had issues
            if pick in [11, 36, 37, 38, 40, 41, 42]:
                continue

            print('\nReading the data from pick No.:', pick)

            file = 'fall21_real_apple_pick' + str(pick)

            # Read Topics from csvs or bagfile
            self.read_topics(file, 'True')

            # Read Variables
            self.extract_variables()

            # Crop the data

            # Filter data
            self.arm_variables_filtered = self.filter_variables(self.arm_variables, 21)
            self.hand_variables_filtered = self.filter_variables(self.hand_variables, 3)

            # Get metadata (e.g. Success or Fail)
            file_label = self.look_at_labels(file)
            print(file_label)

            # Save features for each pick
            if file_label == '(Successful-Pick)':
                # 1 ARM Variables
                # 1.1   Net - Force
                # 1.1.1.    Max Values
                self.max_success_grasp.append(max(self.arm_variables_grasp_filtered[0]))
                self.max_success_pick.append(max(self.arm_variables_pick_filtered[0]))
                # 1.1.2.    Mean Forces
                self.mean_success_grasp.append(st.mean(self.arm_variables_grasp_filtered[0]))
                self.mean_success_pick.append(st.mean(self.arm_variables_pick_filtered[0]))
                # 1.1.3.    Median Forces
                self.median_success_grasp.append(st.median(self.arm_variables_grasp_filtered[0]))
                self.median_success_pick.append(st.median(self.arm_variables_pick_filtered[0]))
                # 1.1.4.    Skewness
                self.force_skew_success_grasp.append(skew(self.arm_variables_grasp_filtered[0]))
                self.force_skew_success_pick.append(skew(self.arm_variables_pick_filtered[0]))
                # 1.1.5.    Area Under Curve
                self.netforce_auc_success_grasp.append(trapz(self.arm_variables_grasp_filtered[0]))
                self.netforce_auc_success_pick.append(trapz(self.arm_variables_pick_filtered[0]))



                # 1.2   Force z
                # 1.2.1.    Max Values
                self.fz_max_success_grasp.append(max(self.arm_variables_grasp_filtered[2]))
                self.fz_max_success_pick.append(max(self.arm_variables_pick_filtered[2]))

                print('The skews are', self.force_skew_success_grasp, self.force_skew_success_pick)

                # 2 HAND Variables
                # 2.1.  Finger 1 acceleration
                # 2.1.1.    Max Values
                self.f1_max_success_grasp.append(max(self.hand_variables_grasp_filtered[0]))
                self.f1_max_success_pick.append(max(self.hand_variables_pick_filtered[0]))
                # 2.2. Finger 1 gyroscope
                self.f1_gyro_auc_success_grasp.append(trapz(self.hand_variables_grasp_filtered[2], dx=0.014))
                self.f1_gyro_auc_success_pick.append(trapz(self.hand_variables_pick_filtered[2], dx=0.014))

            else:
                # MAX Forces
                self.max_failed_grasp.append(max(self.arm_variables_grasp_filtered[0]))
                self.max_failed_pick.append(max(self.arm_variables_pick_filtered[0]))
                # Mean Forces
                self.mean_failed_grasp.append(st.mean(self.arm_variables_grasp_filtered[0]))
                self.mean_failed_pick.append(st.mean(self.arm_variables_pick_filtered[0]))
                # Median Forces
                self.median_failed_grasp.append(st.median(self.arm_variables_grasp_filtered[0]))
                self.median_failed_pick.append(st.median(self.arm_variables_pick_filtered[0]))
                # 1.1.4.    Skewness
                self.force_skew_failed_grasp.append(skew(self.arm_variables_grasp_filtered[0]))
                self.force_skew_failed_pick.append(skew(self.arm_variables_pick_filtered[0]))
                # 1.1.5.    Area Under Curve
                self.netforce_auc_failed_grasp.append(trapz(self.arm_variables_grasp_filtered[0]))
                self.netforce_auc_failed_pick.append(trapz(self.arm_variables_pick_filtered[0]))


                # 1.2   Force z
                # 1.2.1.    Max Values
                self.fz_max_failed_grasp.append(max(self.arm_variables_grasp_filtered[2]))
                self.fz_max_failed_pick.append(max(self.arm_variables_pick_filtered[2]))

                print('The skews are', self.force_skew_failed_grasp, self.force_skew_failed_pick)

                # MAX f1 acc
                self.f1_max_failed_grasp.append(max(self.hand_variables_grasp_filtered[0]))
                self.f1_max_failed_pick.append(max(self.hand_variables_pick_filtered[0]))
                # 2.2. Finger 1 gyroscope
                self.f1_gyro_auc_failed_grasp.append(trapz(self.hand_variables_grasp_filtered[2], dx=0.014))
                self.f1_gyro_auc_failed_pick.append(trapz(self.hand_variables_pick_filtered[2], dx=0.014))

                # print('The aucs were', self.f1_gyro_auc_failed_grasp, self.f1_gyro_auc_failed_pick)

    def read_topics(self, file, csvs_available='True'):

        # Read each topic
        # Note: If the csvs from bagfiles are already there, then there is no need to read bagfile, only csv.
        # This is important because it consumes time (specially the ones sampled at 500Hz)
        start_reading_topics = time.time()
        # print('Start reading topics at: ', start_reading_topics)

        if csvs_available == False:
            # In this case it has to read the bagfiles and extract the csvs
            b = bagreader(self.location + '/' + file + '.bag')

            # a. Arm's Wrench topic: forces and torques
            data_arm_wrench = b.message_by_topic('wrench')
            self.arm_wrench = pd.read_csv(data_arm_wrench)

            # b. Arm's Joint_States topic: angular positions of the 6 joints
            data_arm_joints = b.message_by_topic('joint_states')
            self.arm_joints = pd.read_csv(data_arm_joints)

            # c. Hand's finger1 imu
            data_f1_imu = b.message_by_topic('/applehand/finger1/imu')
            self.f1_imu = pd.read_csv(data_f1_imu)

            # d. Hand's finger1 joint state
            data_f1_joint = b.message_by_topic('/applehand/finger1/jointstate')
            self.f1_state = pd.read_csv(data_f1_joint)

            # e. Hand's finger2 imu
            data_f2_imu = b.message_by_topic('/applehand/finger2/imu')
            self.f2_imu = pd.read_csv(data_f2_imu)

            # f. Hand's finger2 joint state
            data_f2_joint = b.message_by_topic('/applehand/finger2/jointstate')
            self.f2_state = pd.read_csv(data_f2_joint)

            # g. Hand's finger3 imu
            data_f3_imu = b.message_by_topic('/applehand/finger3/imu')
            self.f3_imu = pd.read_csv(data_f3_imu)

            # h. Hand's finger3 joint state
            data_f3_joint = b.message_by_topic('/applehand/finger3/jointstate')
            self.f3_state = pd.read_csv(data_f3_joint)

            # i. Trial events
            data_trial_events = b.message_by_topic('/apple_trial_events')
            self.trial_events = pd.read_csv(data_trial_events)

        else:
            # a. Arm's Wrench topic: forces and torques
            topic = '/rench.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.arm_wrench = pd.read_csv(csv_from_bag)

            # b. Arm's Joint_States topic: angular positions of the 6 joints
            topic = '/oint_states.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.arm_joints = pd.read_csv(csv_from_bag)

            # c. Hand's finger1 imu
            topic = '/applehand-finger1-imu.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.f1_imu = pd.read_csv(csv_from_bag)

            # d. Hand's finger1 joint state
            topic = '/applehand-finger1-jointstate.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.f1_state = pd.read_csv(csv_from_bag)

            # e. Hand's finger2 imu
            topic = '/applehand-finger2-imu.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.f2_imu = pd.read_csv(csv_from_bag)

            # f. Hand's finger2 joint state
            topic = '/applehand-finger2-jointstate.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.f2_state = pd.read_csv(csv_from_bag)

            # g. Hand's finger3 imu
            topic = '/applehand-finger3-imu.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.f3_imu = pd.read_csv(csv_from_bag)

            # h. Hand's finger3 joint state
            topic = '/applehand-finger3-jointstate.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.f3_state = pd.read_csv(csv_from_bag)

            # EVENT's topic
            # i. Trial events
            topic = '/apple_trial_events.csv'
            csv_from_bag = self.location + '/' + file + topic
            self.trial_events = pd.read_csv(csv_from_bag)

        finish_reading_topics = time.time()
        # print('Finished at: ', finish_reading_topics)
        # print('Elapsed time:', (finish_reading_topics - start_reading_topics))


if __name__ == '__main__':
    apples = Apples()

    # --- Step 1: Read data
    apples.read_data(77)

    # # --- Step 2: Create pdfs
    # apples.plot_pdfs()

    # --- Step 3: Subtract features
    apples.plot_features(apples)

    plt.show()



