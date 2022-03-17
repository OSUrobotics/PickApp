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
# import bagpy
# from bagpy import bagreader
# ... Math related packages
import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter, median_filter
# ... Plot related packages
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import seaborn as sns
# sns.set()  # Setting seaborn as default style even if use only matplotlib

plt.close('all')


class PdfPlots:

    def __init__(self):
        # location = 'C:/Users/15416/PycharmProjects/PickApp/data_postprocess1 (only grasp)/'
        self.main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
        self.dataset = '5_real_fall21_x1/'
        self.subfolder = 'PICK/pp1_split/'

    def plots_in_pdf(self, dataset, case):

        for dataset in datasets:
            print('\nDataset: ', dataset)

            for case in cases:

                print('Case: ', case)

                pick_location = main + dataset + '/' + stage + '/' + subfolder + '/' + case + '/'

                # --- Create PDF ---
                pdf_pages = PdfPages(dataset + '_' + stage + '_' + case + '.pdf')
                plot_number = 1
                nb_plots_per_page = 4
                nb_plots = 70

                # Step 1: Create plot array
                f, axrray = plt.subplots(nb_plots_per_page, 1, figsize=(8.5, 14), dpi=100, sharey=True)
                plt.subplots_adjust(left=0.075, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.175)

                for file in tqdm(sorted(os.listdir(pick_location))):

                    # Turn csv into pandas - dataframe
                    df = pd.read_csv(pick_location + file)
                    df = np.array(df)

                    # Read the initial readings
                    initial_fx = df[0, 1]
                    initial_fy = df[0, 2]
                    initial_fz = df[0, 3]
                    initial_time = df[0, 0]

                    # Subtract initial reading to all channels to ease comparison
                    time = df[:, 0] - initial_time
                    fx = df[:, 1] - initial_fx
                    fy = df[:, 2] - initial_fy
                    fz = df[:, 3] - initial_fz

                    # Variables to plot
                    variables = [fx, fy, fz]
                    legends = ['force x', 'force y', 'force z']
                    colors = ['r', 'g', 'b']

                    # Plot variables
                    pos = (plot_number - 1) % 4
                    # print(pos)
                    ax = axrray[pos]

                    y_max = []
                    for i in range(len(variables)):
                        ax.plot(time, variables[i], colors[i], label=legends[i])
                        # axrray[0, 0].plot(variables[i])
                        # axrray[0, 1].plot(variables[i])
                        y_max.append(max(variables[i]))

                    ax.legend()
                    ax.grid()
                    label = case
                    ax.set_title(file + ' ' + f'$\\bf{label}$', size=8, loc='left')
                    ax.set_ylim(-20, 35)
                    ax.set_xlim(0, 5)
                    ax.set_xlabel('Elapsed time [sec]')


                    plot_number += 1

                    # --- Create a new page if the number of rows has been reached
                    if (plot_number - 1) % nb_plots_per_page == 0 or (plot_number - 1) == nb_plots:
                        # print('Page saved\n')
                        pdf_pages.savefig(f)
                        f, axrray = plt.subplots(nb_plots_per_page, 1, figsize=(8.5, 14), dpi=100, sharey=True)
                        # Margins
                        plt.subplots_adjust(left=0.075, bottom=0.05, right=0.925, top=0.95, wspace=0.1, hspace=0.175)

                pdf_pages.savefig(f)
                pdf_pages.close()



if __name__ == "__main__":

    # Step 1 - Open csv
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'

    datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1']

    subfolder = '__for_proxy_real_comparison'

    stage = 'PICK'
    cases = ['success', 'failed']






