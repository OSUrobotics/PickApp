
import csv
import os
import ast
import math
import numpy as np

def get_noise(location):
    noise = []
    for file in os.listdir(location):

        rows = []
        with open(location + file) as csv_file:
            # Create  a csv object
            csv_reader = csv.reader(csv_file, delimiter=',')
            # Extract each data row one by one
            for row in csv_reader:
                rows.append(row)

            # print(rows[1][10])
            noise.append(rows[1][16])

    return noise


if __name__ == "__main__":

    location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/3_proxy_winter22_x1/metadata/'

    x = get_noise(location)

    x_noises = []
    y_noises = []
    z_noises = []
    r_noises = []
    p_noises = []
    yw_noises = []
    for blah in x:
        b = ast.literal_eval(blah)
        c = list(b)
        x_noises.append(c[0])
        y_noises.append(c[1])
        z_noises.append(c[2])
        r_noises.append(c[3] * 180 / math.pi)
        p_noises.append(c[4] * 180 / math.pi)
        yw_noises.append(c[5]* 180 /math.pi)

    print(" --- Percentiles ---- ")
    print(np.percentile(x_noises,[25,50,75]))
    print(np.percentile(y_noises,[25,50,75]))
    print(np.percentile(z_noises,[25,50,75]))
    print(np.percentile(r_noises,[25,50,75]))
    print(np.percentile(p_noises,[25,50,75]))
    print(np.percentile(yw_noises,[25,50,75]))

    print(" --- Mean & Std ---- ")
    print(np.mean(x_noises), np.std(x_noises))
    print(np.mean(y_noises), np.std(y_noises))
    print(np.mean(z_noises), np.std(z_noises))
    print(np.mean(r_noises), np.std(r_noises))
    print(np.mean(p_noises), np.std(p_noises))
    print(np.mean(yw_noises), np.std(yw_noises))



