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
case = 'failed'
similar_pics = [
                       ['4-10', '27'],
                       ['74-6', '5'],
                       ['15-7', '24'],
                       ['15-12', '13'],
                       ['32-4', '57']
                      ]

# case = 'success'
# similar_pics = [
#                            ['11-10', '64'],
#                            ['49-10', '48'],
#                            ['26-12', '71'],
#                            ['25-0', '77'],
#                            ]


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

    plt.plot(proxy_pic_time, proxy_pic_values, '--', label='Apple Proxy')
    plt.plot(real_pic_time, real_pic_values, label='Real Tree')
    plt.grid()
    plt.legend()
    plt.xlabel('Time [sec]')
    plt.ylabel('Force z [N]')
    plt.ylim(-20, 35)
    plt.xlim(0, 4)

    plt.show()

