# @Time : 2/5/2022 12:45 PM
# @Author : Alejandro Velasquez

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# Location

main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1']
stage = 'PICK'
case = 'failed'
subfolder = '__for_proxy_real_comparison'

# Similar Picks: Place Proxy Pics in column 0, and Real Pics in column 1
# case = 'failed'
# types = ['f1', 'f2', 'f3', 'f4']
# similar_pics = [
#                 ['4-10', '27'],  # Hunchback
#                 ['15-7', '24'],  # Pyramid
#                 ['32-4', '57'],   # Whale
#                 ['74-6', '5']   # Mouth
#                 ['15-12', '13'], # Rounded Tip
                  # ]

case = 'success'
types = ['s1', 's2', 's3', 's4']
similar_pics = [
               ['11-10', '64'],     # Right Triangle
               ['49-10', '48'],     # Hunchback Cut
               ['26-12', '71'],     # Mouth Cut
               ['25-0', '77'],      # Long Hunchback
               ]

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
    ax.set_title(types[col])

    col += 1


# Only have the outer labels
for ax in f.get_axes():
    ax.label_outer()

f.suptitle('Successful apple-pick cases')
plt.show()

