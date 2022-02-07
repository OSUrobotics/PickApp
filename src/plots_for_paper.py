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
    f, axrray = plt.subplots(1, len(similar_pics))

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
        ax.plot(proxy_pic_time, proxy_pic_values, '-', label='Apple Proxy', linewidth=2)
        ax.plot(real_pic_time, real_pic_values, label='Real Tree')
        ax.grid()
        ax.legend()
        ax.set_ylim(-5, 30)
        ax.set_xlim(0, 4)
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Force z [N]')
        ax.set_title(titles[col])

        col += 1

    # Only have the outer labels
    for ax in f.get_axes():
        ax.label_outer()

    f.suptitle(case + ' apple-pick cases')
    plt.show()


def auc(x, y):
    # Feature: Area under the curve
    auc = 0
    previous_time = 0
    for i, j in zip(x, y):
        current = j * (i - previous_time)
        auc = auc + current
        previous_time = i

    return auc


def slope(x, y):
    # Feature: Slope

    y = list(y)
    x = list(x)
    delta_y = max(y)
    x_end_index = y.index(delta_y)
    x_end = x[x_end_index]

    x_init = 0
    for i, j in zip(x, y):
        if j > 1:
            x_init = i
            break

    delta_x = x_end - x_init
    slope = (delta_y / delta_x)
    # print(slope)
    return slope




if __name__ == "__main__":

    # Data Location
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1']
    stage = 'PICK'
    subfolder = '__for_proxy_real_comparison'

    # # --- Qualitative Comparison of Real vs Proxy ---
    # # Similar Failed Picks: Place Proxy Pics in column 0, and Real Pics in column 1
    # similar_pics = [
    #                 ['4-10', '27'],  # Hunchback
    #                 ['15-7', '24'],  # Pyramid
    #                 ['32-4', '57'],   # Whale
    #                 ['74-6', '5']   # Mouth
    #                 # ['15-12', '13'], # Rounded Tip
    #                   ]
    # qual_compare(main, datasets, stage, subfolder, 'failed', ['f1', 'f2', 'f3', 'f4'], similar_pics)
    #
    # # Similar Successful Pics
    # similar_pics = [
    #                ['11-10', '64'],     # Right Triangle
    #                ['49-10', '48'],     # Hunchback Cut
    #                ['26-12', '71'],     # Mouth Cut
    #                ['25-0', '77'],      # Long Hunchback
    #                ]
    # qual_compare(main, datasets, stage, subfolder, 'success', ['s1', 's2', 's3', 's4'], similar_pics)

    # --- Quantitative Comparison of Real vs Proxy ---
    # Step 1 - Get the list
    case = 'success'
    # case = 'failed'
    proxy_files_location = main + datasets[0] + '/' + stage + '/' + subfolder + '/' + case + '/'
    real_files_location = main + datasets[1] + '/' + stage + '/' + subfolder + '/' + case + '/'

    rpeak_values = []
    raucs = []
    rslopes = []
    rskews = []
    for file in os.listdir(real_files_location):
        # print(file)
        time, values = pic_list(real_files_location + file, 3)
        # Get Peak Value
        peak_value = max(values)

        # Area under the curve
        fz_auc = auc(time, values)

        # Slope
        s = slope(time, values)

        # Skewness
        rskews.append(skew(values))

        rpeak_values.append(peak_value)
        raucs.append(fz_auc)
        rslopes.append(s)

    d = {'Force z': rpeak_values, 'AUC': raucs, 'Slope': rslopes, 'Skew': rskews}
    df1 = pd.DataFrame(data=d)

    # Step 2 - Get the list
    ppeak_values = []
    paucs = []
    pslopes = []
    pskews = []
    for file in os.listdir(proxy_files_location):
        time, values = pic_list(proxy_files_location + file, 3)
        # Feature to extract
        peak_value = max(values)

        # Area under the curve
        fz_auc = auc(time, values)

        # Slope
        s = slope(time, values)

        pskews.append(skew(values))

        ppeak_values.append(peak_value)
        paucs.append(fz_auc)
        pslopes.append(s)

    d = {'Force z': ppeak_values, 'AUC': paucs, 'Slope': pslopes, 'Skew': pskews}
    df2 = pd.DataFrame(data=d)

    df3 = pd.concat([df1, df2], axis=1, keys=['real', 'proxy']).stack(0)
    df3 = df3.reset_index(level=1)
    # bplot = sns.boxplot(x='level_1', y='AUC', data=df3, palette='colorblind')
    bplot = sns.swarmplot(x='level_1', y='Skew', data=df3, color ='black')

    plt.title('Skew ' + case + ' picks')
    plt.grid()
    plt.ylim(-2, 5)
    plt.show()


