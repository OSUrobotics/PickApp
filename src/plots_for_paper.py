# @Time : 2/5/2022 12:45 PM
# @Author : Alejandro Velasquez
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy.stats import skew

from matplotlib.patches import PathPatch


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """
    # https://stackoverflow.com/questions/56838187/how-to-create-spacing-between-same-subgroup-in-seaborn-boxplot

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def pic_list(file, variable):

    # Turn csv into pandas - dataframe
    df = pd.read_csv(file)
    df = np.array(df)

    # Read the initial readings
    initial_value = df[0, variable]
    initial_time = df[0, 0]

    # Subtract initial reading to all channels to ease comparison
    time = df[:, 0] - initial_time
    value = df[:, variable] - initial_value

    return time, value


def qual_compare(main, datasets, stage, subfolder, case, titles, similar_pics):
    """
    Creates a plot of the time series of the given similar -pics in order to compare them visually
    :param main:
    :param datasets:
    :param stage:
    :param subfolder:
    :param case:
    :param titles:
    :param similar_pics:
    :return:
    """
    # f = plt.figure(figsize=(4, 4))
    f, axrray = plt.subplots(1, len(similar_pics), figsize=(12, 4))

    col = 0
    for couple in similar_pics:
        proxy_pic_number = couple[0]
        real_pic_number = couple[1]

        # Concatenate file name
        proxy_pic_file = 'apple_proxy_pick' + proxy_pic_number + '_pick_wrench.csv'
        real_pic_file = 'real_apple_pick_' + real_pic_number + '_pick_wrench.csv'

        # Concatenate location
        proxy_file_location = main + datasets[0] + '/' + stage + '/' + subfolder + '/' + case + '/' + proxy_pic_file
        real_file_location = main + datasets[1] + '/' + stage + '/' + subfolder + '/' + case + '/' + real_pic_file

        # Get simplified lists with the variable of interest
        proxy_pic_time, proxy_pic_values = pic_list(proxy_file_location, 3)
        real_pic_time, real_pic_values = pic_list(real_file_location, 3)

        # Pot
        ax = axrray[col]
        # ax.plot(proxy_pic_time, proxy_pic_values, '-', label='Apple Proxy', linewidth=2)
        ax.plot(real_pic_time, real_pic_values, label='Real Tree')
        # ax.grid()
        ax.legend()
        ax.set_ylim(-5, 25)
        ax.set_xlim(0, 3)
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Force z [N]')
        ax.set_title(titles[col])

        col += 1

    # Only have the outer labels
    for ax in f.get_axes():
        ax.label_outer()

    # f.suptitle(case + ' apple-pick cases')


def auc(x, y, start_idx, end_idx):

    # Feature: Area under the curve
    auc = 0

    # Approach 1:
    # previous_time = 0
    # for i in range(start_idx, end_idx):
    #     print('i=', i)
        # current = math.sqrt(y[i]**2) * (x[i]-previous_time)
        # auc = auc + current
        # previous_time = x[i]

    # Approach 2: Built-in Function
    y_list = y[start_idx:end_idx]
    x_list = x[start_idx: end_idx]
    auc = np.trapz(y_list, x_list)

    return auc


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
        x_end = x[-1]
        x_init_idx = 0
        x_end_idx = len(x) - 1

    elif len(tc) >= 2:
        # In two or more crossings are detected, take the first one as the initial, and the second as the final
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
            print('**************************************************************')
            tcb.append(x_init + 1)
            tcb_idx.append(len(x)-1)

        x_end = tcb[1]
        x_end_idx = tcb_idx[1]

    print('Start at %.2f and ends at %.2f' % (x_init, x_end))
    return x_init, x_end, x_init_idx, x_end_idx


def shape_of_pick(x, y, t_start, t_end):
    # Do a cool check of the shape

    # Peak value
    y = list(y)
    delta_y = max(y)

    # Time when peak value happens
    x = list(x)
    x_max_index = y.index(delta_y)
    x_max = x[x_max_index]

    ratio = (x_max - t_start) / (t_end - t_start)

    if ratio <= 0.4:
        shape = "Start"
    elif 0.4 < ratio <= 0.6:
        shape = "Middle"
    elif 0.6 < ratio <= 0.85:
        shape = "Whale"
    elif 0.85 < ratio <= 1:
        shape = "End"
    else:
        shape = "Other"

    return shape


def slope(x, y, t_start, t_end):

    # Peak Value
    y = list(y)
    delta_y = max(y)

    # Time when the peak happens
    x = list(x)
    x_max_index = y.index(delta_y)
    x_max = x[x_max_index]

    # Slope
    delta_x = x_max - t_start
    slope = (delta_y / delta_x)

    return slope


def strip_and_box(dataframe, feature, case):

    fig = plt.figure(figsize=(4, 4))

    # OSU color RGB (241, 93, 34), hex (#F15D22)
    if case == 'failed':
        my_pal = {"Real Tree": "#FF0000", "Apple Proxy": "#FF7C80"}
    else:
        my_pal = {"Real Tree": "#387A23", "Apple Proxy": "#A9D18E"}

    # bplot = sns.boxplot(x='level_1', y=feature, data=df3, palette='colorblind', width=0.4, boxprops=dict(alpha=.9))
    bplot = sns.boxplot(x='level_1', y=feature, data=dataframe, palette=my_pal, width=0.4, boxprops=dict(alpha=.8))
    bplot = sns.stripplot(x='level_1', y=feature, data=dataframe, color='black', size=4, alpha=.6)
    # bplot = sns.stripplot(x='level_1', y=col, data=df3)

    # plt.title(col + ' ' + case + ' picks' + '(considering diameter)')
    # plt.title(col + ' ' + case + ' picks' + " (all except " + shape + ")")
    plt.title(feature + ' / ' + case + ' picks')
    # plt.title(col + ' ' + case + ' picks' + " ( " + shape + ")")
    # plt.title(col + ' ' + case + ' picks' + " (all)")

    plt.grid()
    plt.ylim(-5, 40)


def count_plot(proxy_picks_shapes, real_picks_shapes, case):
    # ... Shape Plots ...
    p_picks_perc = []
    for number in proxy_picks_shapes:
        p_picks_perc.append(number / sum(proxy_picks_shapes))

    r_picks_perc = []
    for number in real_picks_shapes:
        r_picks_perc.append(number / sum(real_picks_shapes))

    print('Proxy shapes', proxy_picks_shapes)
    print('Real pick shapes', real_picks_shapes)
    print("Proxy pick shapes  %.2f, %.2f, %.2f, %.2f, %.2f" % (p_picks_perc[0], p_picks_perc[1], p_picks_perc[2], p_picks_perc[3], p_picks_perc[4]))
    print("Real pick shapes  %.2f, %.2f, %.2f, %.2f, %.2f" % (r_picks_perc[0], r_picks_perc[1], r_picks_perc[2], r_picks_perc[3], r_picks_perc[4]))

    df = pd.DataFrame({"Percentage": p_picks_perc + r_picks_perc,
                      "Domain": ['Proxy', 'Proxy', 'Proxy', 'Proxy', 'Proxy', 'Real', 'Real', 'Real', 'Real', 'Real'],
                      "Shape": ['End', 'L-Skew', 'Middle', 'R-Skew', 'Other', 'End', 'L-Skew', 'Middle', 'R-Skew', 'Other']})

    fig = plt.figure(figsize=(4, 4))
    sns.barplot(x="Shape", y="Percentage", data=df, hue="Domain")
    plt.grid()
    plt.title(case)


if __name__ == "__main__":

    # Data Location
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1', '1_proxy_rob537_x1']
    stage = 'PICK'
    subfolder = '__for_proxy_real_comparison'

    # # --- Qualitative Comparison of Real vs Proxy ---
    # # Similar Failed Picks: Place Proxy Pics in column 0, and Real Pics in column 1
    # similar_pics = [
    #                 ['4-10', '27'],  # Hunchback
    #                 ['15-7', '24'],  # Pyramid
    #                 ['32-4', '57'],   # Whale
    #                 ['74-6', '5']   # Mouth
    # #                 ['15-12', '13'], # Rounded Tip
    #                   ]
    similar_pics = [
        ['4-10', '49'],  # Hunchback
        ['15-7', '24'],  # Pyramid
        ['32-4', '18'],  # Whale
        ['74-6', '17']  # Mouth
        #                 ['15-12', '13'], # Rounded Tip
        ]

    qual_compare(main, datasets, stage, subfolder, 'failed', ['Start', 'Middle', 'Whale', 'End'], similar_pics)
    #
    # # Similar Successful Pics
    # similar_pics = [
    #                ['11-10', '64'],     # Right Triangle
    #                ['49-10', '48'],     # Hunchback Cut
    #                ['26-12', '71'],     # Mouth Cut
                   # ['25-0', '77'],      # Long Hunchback
                   # ]
    # qual_compare(main, datasets, stage, subfolder, 'success', ['s1', 's2', 's3', 's4'], similar_pics)

    # --- Quantitative Comparison of Real vs Proxy ---
    # Step 1 - Get the list
    # case = 'success'
    case = 'failed'

    included_shapes = ['Start', 'Middle', 'Whale', 'End']
    # included_shapes = ['Other']
    # included_shapes = ['Start']
    # included_shapes = ['End']

    proxy_files_location = main + datasets[0] + '/' + stage + '/' + subfolder + '/' + case + '/'
    # proxy_files_location_b = main + datasets[2] + '/' + stage + '/' + subfolder + '/' + case + '/'
    real_files_location = main + datasets[1] + '/' + stage + '/' + subfolder + '/' + case + '/'

    rpeak_values = []
    raucs = []
    rslopes = []

    r_end_shape, r_middle_shape, r_start_shape, r_other_shape, r_whale_shape = 0, 0, 0, 0, 0

    counter =0
    for file in os.listdir(real_files_location):
        print('\n', file)

        # Subtract the pick number from the name
        name = str(file)
        start = name.index('k')
        end = name.index('w')
        pick_number = name[start+2: end - 6]
        # print(pick_number)

        # Get the time series
        time, values = pic_list(real_files_location + file, 3)

        # Get some common features
        start, end, start_idx, end_idx = crossings(time, values)        # Start and ending time of the force plot
        peak_value = max(values)                                        # Peak Value
        fz_auc = auc(time, values, start_idx, end_idx)                  # Area under the curve
        s = slope(time, values, start, end)                             # Slope
        type = shape_of_pick(time, values, start, end)                  # Shape

        # Check type
        print(type)
        if type == 'End':
            r_end_shape += 1
        elif type == 'Middle':
            r_middle_shape += 1
        elif type == 'Whale':
            r_whale_shape += 1
        elif type == 'Other':
            r_other_shape += 1
        elif type == 'Start':
            r_start_shape += 1

        # Save lists
        if type in included_shapes:
            counter +=1
            rpeak_values.append(peak_value)
            raucs.append(fz_auc)
            rslopes.append(s)

    real_picks_shapes = [r_end_shape, r_whale_shape, r_middle_shape, r_start_shape, r_other_shape]

    d = {'Force z - Peak [N]': rpeak_values, 'Force z - AUC [N.s]': raucs, 'Force z - Slope [N/s]': rslopes}
    df1 = pd.DataFrame(data=d)


    # Step 2 - Get the list
    ppeak_values = []
    paucs = []
    pslopes = []

    p_end_shape, p_middle_shape, p_start_shape, p_other_shape, p_whale_shape = 0, 0, 0, 0, 0

    for file in os.listdir(proxy_files_location):
        print('\n', file)
        time, values = pic_list(proxy_files_location + file, 3)

        # Get some common features
        start, end, start_idx, end_idx = crossings(time, values)    # Start and ending time of the force plot
        peak_value = max(values)                                    # Peak Value
        fz_auc = auc(time, values, start_idx, end_idx)              # Area under the curve
        s = slope(time, values, start, end)                         # Slope
        type = shape_of_pick(time, values, start, end)              # Shape

        # Check type
        print(type)
        if type == 'End':
            p_end_shape += 1
            # print(p_end_shape)
        elif type == 'Middle':
            p_middle_shape += 1
        elif type == 'Whale':
            p_whale_shape += 1
        elif type == 'Other':
            p_other_shape += 1
        elif type == 'Start':
            p_start_shape += 1

        # Save lists
        if type in included_shapes:
            ppeak_values.append(peak_value)
            paucs.append(fz_auc)
            pslopes.append(s)


    proxy_picks_shapes = [p_end_shape, p_whale_shape, p_middle_shape, p_start_shape, p_other_shape]

    d = {'Force z - Peak [N]': ppeak_values, 'Force z - AUC [N.s]': paucs, 'Force z - Slope [N/s]': pslopes}
    df2 = pd.DataFrame(data=d)

    df3 = pd.concat([df1, df2], axis=1, keys=['Real Tree', 'Apple Proxy']).stack(0)
    df3 = df3.reset_index(level=1)

    # ... Plots ...
    strip_and_box(df3, 'Force z - Peak [N]', case)
    strip_and_box(df3, 'Force z - AUC [N.s]', case)
    strip_and_box(df3, 'Force z - Slope [N/s]', case)
    count_plot(proxy_picks_shapes, real_picks_shapes, case)


    plt.show()

