# @Time : 2/5/2022 12:45 PM
# @Author : Alejandro Velasquez
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats
from scipy.stats import linregress, skew
from scipy.signal import cwt, find_peaks_cwt, ricker, welch
from collections import defaultdict
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from matplotlib.patches import PathPatch


# Function from ts-fresh
def autocorrelation(x, lag):
    """
    Calculates the autocorrelation of the specified lag, according to the formula [1]

    .. math::

        \\frac{1}{(n-l)\\sigma^{2}} \\sum_{t=1}^{n-l}(X_{t}-\\mu )(X_{t+l}-\\mu)

    where :math:`n` is the length of the time series :math:`X_i`, :math:`\\sigma^2` its variance and :math:`\\mu` its
    mean. `l` denotes the lag.

    .. rubric:: References

    [1] https://en.wikipedia.org/wiki/Autocorrelation#Estimation

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    # This is important: If a series is passed, the product below is calculated
    # based on the index, which corresponds to squaring the series.
    if isinstance(x, pd.Series):
        x = x.values
    if len(x) < lag:
        return np.nan
    # Slice the relevant subseries based on the lag
    y1 = x[: (len(x) - lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(x)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(x)
    if np.isclose(v, 0):
        return np.NaN
    else:
        return sum_product / ((len(x) - lag) * v)


def cwt_coefficients(x):
    """
    Calculates a Continuous wavelet transform for the Ricker wavelet, also known as the "Mexican hat wavelet" which is
    defined by

    .. math::
        \\frac{2}{\\sqrt{3a} \\pi^{\\frac{1}{4}}} (1 - \\frac{x^2}{a^2}) exp(-\\frac{x^2}{2a^2})

    where :math:`a` is the width parameter of the wavelet function.

    This feature calculator takes three different parameter: widths, coeff and w. The feature calculator takes all the
    different widths arrays and then calculates the cwt one time for each different width array. Then the values for the
    different coefficient for coeff and width w are returned. (For each dic in param one feature is returned)

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"widths":x, "coeff": y, "w": z} with x array of int and y,z int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    calculated_cwt = {}
    res = []
    indices = []

    # param = {"widths": [2,5,10,20], "coeff": 11, "w": 2}
    widths = (2,5,10,20)
    w = 5
    coeff = 14

    # widths = tuple(parameter_combination["widths"])
    # w = parameter_combination["w"]
    # coeff = parameter_combination["coeff"]

    if widths not in calculated_cwt:
        calculated_cwt[widths] = cwt(x, ricker, widths)

    calculated_cwt_for_widths = calculated_cwt[widths]

    indices += ["coeff_{}__w_{}__widths_{}".format(coeff, w, widths)]

    i = widths.index(w)
    if calculated_cwt_for_widths.shape[1] <= coeff:
        res += [np.NaN]
    else:
        res += [calculated_cwt_for_widths[i, coeff]]

    # return zip(indices, res)
    return res


def _aggregate_on_chunks(x, f_agg, chunk_len):
    """
    Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len

    :param x: the time series to calculate the aggregation of
    :type x: numpy.ndarray
    :param f_agg: The name of the aggregation function that should be an attribute of the pandas.Series
    :type f_agg: str
    :param chunk_len: The size of the chunks where to aggregate the time series
    :type chunk_len: int
    :return: A list of the aggregation function over the chunks
    :return type: list
    """
    return [
        getattr(x[i * chunk_len : (i + 1) * chunk_len], f_agg)()
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]


def agg_linear_trend(x, chunk):
    # Source: https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#agg_linear_trend
    """
    Calculates a linear least-squares regression for values of the time series that were aggregated over chunks versus
    the sequence from 0 up to the number of chunks minus one.

    This feature assumes the signal to be uniformly sampled. It will not use the time stamps to fit the model.

    The parameters attr controls which of the characteristics are returned. Possible extracted attributes are "pvalue",
    "rvalue", "intercept", "slope", "stderr", see the documentation of linregress for more information.

    The chunksize is regulated by "chunk_len". It specifies how many time series values are in each chunk.

    Further, the aggregation function is controlled by "f_agg", which can use "max", "min" or , "mean", "median"

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param param: contains dictionaries {"attr": x, "chunk_len": l, "f_agg": f} with x, f an string and l an int
    :type param: list
    :return: the different feature values
    :return type: pandas.Series
    """

    calculated_agg = defaultdict(dict)
    res_data = []
    res_index = []

    # for parameter_combination in param:
    # print(param)
    # print("\nParam Combi:", parameter_combination)

    # chunk_len = parameter_combination['chunk_len']
    # f_agg = parameter_combination["f_agg"]

    attr = 'rvalue'
    chunk_len = chunk
    f_agg = 'mean'

    # attr = 'intercept'
    # chunk_len = 5
    # f_agg = 'min'


    if f_agg not in calculated_agg or chunk_len not in calculated_agg[f_agg]:
        if chunk_len >= len(x):
            calculated_agg[f_agg][chunk_len] = np.NaN
        else:
            aggregate_result = _aggregate_on_chunks(x, f_agg, chunk_len)
            lin_reg_result = linregress(
                range(len(aggregate_result)), aggregate_result
            )
            calculated_agg[f_agg][chunk_len] = lin_reg_result

    # attr = parameter_combination["attr"]

    if chunk_len >= len(x):
        res_data.append(np.NaN)
    else:
        res_data.append(getattr(calculated_agg[f_agg][chunk_len], attr))

    res_index.append(
        'attr_"{}"__chunk_len_{}__f_agg_"{}"'.format(attr, chunk_len, f_agg)
    )

    # return zip(res_index, res_data)
    return res_data[0]


def cid_ce(x, normalize):
    """
    This function calculator is an estimate for a time series complexity [1] (A more complex time series has more peaks,
    valleys etc.). It calculates the value of

    .. math::

        \\sqrt{ \\sum_{i=1}^{n-1} ( x_{i} - x_{i-1})^2 }

    .. rubric:: References

    |  [1] Batista, Gustavo EAPA, et al (2014).
    |  CID: an efficient complexity-invariant distance for time series.
    |  Data Mining and Knowledge Discovery 28.3 (2014): 634-669.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param normalize: should the time series be z-transformed?
    :type normalize: bool

    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s != 0:
            x = (x - np.mean(x)) / s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))


def kurtosis(x):
    """
    Returns the kurtosis of x (calculated with the adjusted Fisher-Pearson standardized
    moment coefficient G2).

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)



# Other Functions

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

    # variable: String

    # Turn csv into pandas - dataframe
    df = pd.read_csv(file)
    # df = np.array(df)

    # print(file)
    # Read the initial readings
    initial_value = df.iloc[1][variable]
    initial_time = df.iloc[0][0]

    # Subtract initial reading to all channels to ease comparison
    # time = df[:, 0] - initial_time
    time = df['# elapsed time'] - initial_time
    # value = df[:, variable] - initial_value
    value = df[variable] - initial_value

    return time, value


def qual_compare(main, datasets, stage, subfolder, case, titles, similar_pics, variable, topic):
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
        proxy_pic_file = 'apple_proxy_pick' + proxy_pic_number + '_pick_' + str(topic) + '.csv'
        real_pic_file = 'real_apple_pick_' + real_pic_number + '_pick_' + str(topic) + '.csv'

        # Concatenate location
        proxy_file_location = main + datasets[0] + '/' + stage + '/' + subfolder + '/' + case + '/' + proxy_pic_file
        real_file_location = main + datasets[1] + '/' + stage + '/' + subfolder + '/' + case + '/' + real_pic_file

        # Get simplified lists with the variable of interest
        proxy_pic_time, proxy_pic_values = pic_list(proxy_file_location, variable)
        real_pic_time, real_pic_values = pic_list(real_file_location, variable)

        # Pot
        ax = axrray[col]
        ax.plot(proxy_pic_time, proxy_pic_values, '-', label='Apple Proxy', linewidth=2)
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
        # print('Flat')
        x_init = x[0]
        x_end = x.iloc[-1]
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
            # print('**************************************************************')
            tcb.append(x_init + 0.5)
            tcb_idx.append(len(x)-1)

        x_end = tcb[1]
        x_end_idx = tcb_idx[1]

    # print('Start at %.2f and ends at %.2f' % (x_init, x_end))
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

    return slope, x_max_index


def strip_and_box(dataframe, feature, case, variable):

    fig = plt.figure(figsize=(3, 2))

    # OSU color RGB (241, 93, 34), hex (#F15D22)
    if case == 'failed':
        my_pal = {"Real Tree": "#FF0000", "Apple Proxy": "#FF7C80"}
    else:
        my_pal = {"Real Tree": "#387A23", "Apple Proxy": "#A9D18E"}

    bplot = sns.boxplot(x='level_1', y=feature, data=dataframe, palette='colorblind', width=0.4, boxprops=dict(alpha=.9))
    # bplot = sns.boxplot(x='level_1', y=feature, data=dataframe, palette=my_pal, width=0.4, boxprops=dict(alpha=.4))
    bplot = sns.stripplot(x='level_1', y=feature, data=dataframe, color='black', size=4, alpha=.6)
    # bplot = sns.stripplot(x='level_1', y=col, data=df3)

    # plt.title(col + ' ' + case + ' picks' + '(considering diameter)')
    # plt.title(col + ' ' + case + ' picks' + " (all except " + shape + ")")
    plt.title(variable + ' / ' + feature + ' / ' + case + ' picks')
    # plt.title(col + ' ' + case + ' picks' + " ( " + shape + ")")
    # plt.title(col + ' ' + case + ' picks' + " (all)")
    plt.ylabel('Aggregated Linear Trend')

    plt.grid()
    plt.ylim(-1, 1.1)


def count_plot(proxy_picks_shapes, real_picks_shapes, case, variable):
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
    plt.title(variable + ' / ' + case)


def statistics(feature):

    # print(feature)
    feature = [x for x in feature if math.isnan(x) == False]
    mean = round(np.mean(feature), 3)
    stdev = round(np.std(feature), 3)
    cv = round(stdev / mean, 3)

    return mean, stdev, cv


def temporal(locations, topic, variable, chunk):

    end_shape, middle_shape, start_shape, other_shape, whale_shape = 0, 0, 0, 0, 0
    peak_values = []
    aucs = []
    slopes = []
    agg_lins = []

    for location in locations:
        for file in sorted(os.listdir(location)):
            if file.endswith(str(topic) + '.csv'):
                # Subtract the pick number from the name
                name = str(file)
                start = name.index('k')
                end = name.index(str(topic))
                pick_number = name[start+2: end - 6]
                # print(pick_number)

                # Get the time series
                time, values = pic_list(location + file, variable)

                # Get some common features
                start, end, start_idx, end_idx = crossings(time, values)        # Start and ending time of the force plot
                peak_value = max(values)                                        # Peak Value
                fz_auc = auc(time, values, start_idx, end_idx)                  # Area under the curve
                s, tmax_idx = slope(time, values, start, end)                             # Slope
                # type = shape_of_pick(time, values, start, end)                  # Shape

                # Other exotic features
                agg_lin = agg_linear_trend(values[start_idx: end_idx], chunk)
                # agg_lin = s

                #agg_lin = agg_linear_trend(values, {"attr": "intercept", "chunk_len": 5, "f_agg": "min"})
                # other = kurtosis(values[start_idx: end_idx])
                # other = cwt_coefficients(values[start_idx: end_idx])
                # other = other[0]
                # other = skew(values[start_idx: end_idx])
                # other = autocorrelation(values, 4)
                # other = linregress(time[tmax_idx-1:end_idx], values[tmax_idx-1:end_idx])
                # other = other[2]
                # other = cid_ce(values[start_idx: end_idx],0)


                # Check the type of the shape
                # print('\n', file)
                # print('El valor del agg es:',agg_lin)

                # if type == 'End':
                #     end_shape += 1
                # elif type == 'Middle':
                #     middle_shape += 1
                # elif type == 'Whale':
                #     whale_shape += 1
                # elif type == 'Other':
                #     other_shape += 1
                # elif type == 'Start':
                #     start_shape += 1

                # Save lists
                if True:
                # if type in included_shapes:
                    peak_values.append(peak_value)
                    aucs.append(fz_auc)
                    slopes.append(s)
                    agg_lins.append(agg_lin)

    shapes_frec = [end_shape, whale_shape, middle_shape, start_shape, other_shape]

    return peak_values, aucs, slopes, agg_lins, shapes_frec


def rfc(X_train_list, y_train_list, X_test_list, y_test_list):

    # ------------ Train Random Forest Classifier -----------
    X_train = np.array(X_train_list)
    # print(len(X_train))
    X = X_train.reshape(-1, 1)
    # print(X_train)

    y_train = np.array(y_train_list)

    clf = MLPClassifier(solver='adam', random_state=None, max_iter=2000, hidden_layer_sizes=50)
    # clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf.fit(X, y_train)

    # ------------- Test it! ---------------------------------
    X_test = np.array(X_test_list)
    X_val = X_test.reshape(-1, 1)
    y_val = np.array(y_test_list)

    performance = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    for j, k in zip(X_val, y_val):
        grasp_prediction = clf.predict([j])
        # print(grasp_prediction)

        if grasp_prediction == k:
            # print("yeahh")
            performance += 1

            if grasp_prediction == 1:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if grasp_prediction == 1:
                false_positives += 1
            else:
                false_negatives += 1

    result = performance / len(X_test)
    print("\nClassifier: Random Forest")
    # print("Parameters: %i tsfresh features" % n_features)
    print("Accuracy: %.2f" % result)
    print("Confusion Matrix")
    print("                  Reference      ")
    print("Prediction    Success     Failure")
    print("   Success       %i          %i" % (true_positives, false_positives))
    print("   Failure       %i          %i" % (false_negatives, true_negatives))
    print('\n')


def simple_classifier(X_train_list, y_train_list, X_test_list, y_test_list):

    success = []
    failed = []
    # Get the mean of the successful
    for x, y in zip(X_train_list, y_train_list):
        if y == 1:
            success.append(x)
        else:
            failed.append(x)

    # Get the means:

    success_mean = np.mean(success)
    failed_mean = np.mean(failed)

    threshold = (success_mean + failed_mean)/2
    print(threshold)

    # Classify
    performance = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    for j, k in zip(X_test_list, y_test_list):

        if j < threshold:
            grasp_prediction = 0
        else:
            grasp_prediction = 1

        # grasp_prediction = clf.predict([j])
        print(grasp_prediction)

        if grasp_prediction == k:
            # print("yeahh")
            performance += 1

            if grasp_prediction == 1:
                true_positives += 1
            else:
                true_negatives += 1
        else:
            if grasp_prediction == 1:
                false_positives += 1
            else:
                false_negatives += 1

    result = performance / len(X_test_list)
    print("\nClassifier: Random Forest")
    # print("Parameters: %i tsfresh features" % n_features)
    print("Accuracy: %.2f" % result)
    print("Confusion Matrix")
    print("                  Reference      ")
    print("Prediction    Success     Failure")
    print("   Success       %i          %i" % (true_positives, false_positives))
    print("   Failure       %i          %i" % (false_negatives, true_negatives))
    print('\n')


if __name__ == "__main__":

    # --- Data Location
    main = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/RAL22 Paper/'
    datasets = ['3_proxy_winter22_x1', '5_real_fall21_x1', '1_proxy_rob537_x1']
    stage = 'PICK'
    subfolder = '__for_proxy_real_comparison'

    # --- Variable that we wish to analyze
    variables = [' force_z', ' f1_acc_z', ' f3_acc_z', ' torque_z']
    variable = variables[0]

    # Assign the topic
    if variable == ' force_z' or variables == ' force_x' or variables == ' force_y' or variable == ' torque_z':
        topic = 'wrench'
    elif variable == ' f1_acc_x' or variable == ' f1_acc_y' or variable == ' f1_acc_z' or variable == ' f1_gyro_x':
        topic = 'f1_imu'
    elif variable == ' f2_acc_x' or variable == ' f2_acc_y' or variable == ' f2_acc_z' or variable == ' f2_gyro_x':
        topic = 'f2_imu'
    elif variable == ' f3_acc_x' or variable == ' f3_acc_y' or variable == ' f3_acc_z' or variable == ' f3_gyro_x':
        topic = 'f3_imu'

    # --- Qualitative Comparison of Real vs Proxy ---
    # Similar Failed Picks: place Proxy Pics in column 0, and Real Pics in column 1
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

    # qual_compare(main, datasets, stage, subfolder, 'failed', ['Start', 'Middle', 'Whale', 'End'], similar_pics, variable, topic)
    #
    # # Similar Successful Pics
    # similar_pics = [
    #                ['11-10', '64'],     # Right Triangle
    #                ['49-10', '48'],     # Hunchback Cut
    #                ['26-12', '71'],     # Mouth Cut
                   # ['25-0', '77'],      # Long Hunchback
                   # ]
    # qual_compare(main, datasets, stage, subfolder, 'success', ['s1', 's2', 's3', 's4'], similar_pics, variable)

    # --- Quantitative Comparison of Real vs Proxy ---
    # Step 1 - Get the list
    # case = 'success'

    included_shapes = ['Start', 'Middle', 'Whale', 'End', 'Other']

    av_list = []
    av_ref = 1
    it = 17    #Chunk size

    # ---------------------------------------------- Failed Cases ------------------------------------------------------
    case = 'failed'

    # Concatenate each folder
    proxy_files_location_a = main + datasets[0] + '/' + stage + '/' + subfolder + '/' + case + '/'
    proxy_files_location_b = main + datasets[2] + '/' + stage + '/' + subfolder + '/' + case + '/'
    real_files_location = main + datasets[1] + '/' + stage + '/' + subfolder + '/' + case + '/'

    # Get features from the REAL PICKS datasets
    rpeak_values, raucs, rslopes, raggs, real_picks_shapes = temporal([real_files_location], topic, variable, it)
    d = {'Peak [N]': rpeak_values, 'AUC [N.s]': raucs, 'Slope [N/s]': rslopes, 'Agg Linear Trend': raggs}
    real_failed_df = pd.DataFrame(data=d)

    # Get features from the PROXY PICKS datasets
    ppeak_values, paucs, pslopes, paggs, proxy_picks_shapes = temporal([proxy_files_location_a, proxy_files_location_b], topic, variable, it)
    d = {'Peak [N]': ppeak_values, 'AUC [N.s]': paucs, 'Slope [N/s]': pslopes, 'Agg Linear Trend': paggs}
    proxy_failed_df = pd.DataFrame(data=d)

    # Perform t-test between real and proxy
    print('T-test for failed picks')
    raggs = [x for x in raggs if math.isnan(x) == False]
    paggs = [x for x in paggs if math.isnan(x) == False]

    failed_val_set = raggs
    failed_train_set = paggs


    s_fa, p_fa = stats.ttest_ind(raggs, paggs)  # Static and P-Values
    print(s_fa, p_fa)

    print("\n**** Real Statistics ****")
    print('          Mean,  Std, Cv')
    # print("Peak : ", statistics(rpeak_values))
    # print("Slope: ", statistics(rslopes))
    # print("AUCS : ", statistics(raucs))
    print("Feature : ", statistics(raggs))

    #
    print("\n**** Proxy Statistics ****")
    print('          Mean,  Std, Cv')
    # print("Peak : ", statistics(ppeak_values))
    # print("Slope: ", statistics(pslopes))
    # print("AUCS : ", statistics(paucs))
    print("Feature : ", statistics(paggs))

    # ... Plots ...
    df3 = pd.concat([real_failed_df, proxy_failed_df], axis=1, keys=['Real Tree', 'Apple Proxy']).stack(0)
    df3 = df3.reset_index(level=1)
    strip_and_box(df3, 'Agg Linear Trend', case, variable)

    # strip_and_box(df3, 'Peak [N]', case, variable)
    # strip_and_box(df3, 'AUC [N.s]', case, variable)

    # strip_and_box(df3, 'Slope [N/s]', case, variable)
    # count_plot(proxy_picks_shapes, real_picks_shapes, case, variable)

    # -------------------------------------------- SUCCESS Cases -------------------------------------------------------
    case = 'success'

    proxy_files_location_a = main + datasets[0] + '/' + stage + '/' + subfolder + '/' + case + '/'
    proxy_files_location_b = main + datasets[2] + '/' + stage + '/' + subfolder + '/' + case + '/'
    real_files_location = main + datasets[1] + '/' + stage + '/' + subfolder + '/' + case + '/'

    # Get the information from the REAL PICKS datasets
    rpeak_values, raucs, rslopes, raggs, real_picks_shapes = temporal([real_files_location], topic, variable, it)
    d = {'Peak [N]': rpeak_values, 'AUC [N.s]': raucs, 'Slope [N/s]': rslopes, 'Agg Linear Trend': raggs}
    real_success_df = pd.DataFrame(data=d)

    # Get the information from the PROXY PICKS datasets
    ppeak_values, paucs, pslopes, paggs, proxy_picks_shapes = temporal([proxy_files_location_a, proxy_files_location_b], topic,
                                                                variable, it)
    d = {'Peak [N]': ppeak_values, 'AUC [N.s]': paucs, 'Slope [N/s]': pslopes, 'Agg Linear Trend': paggs}
    proxy_success_df = pd.DataFrame(data=d)


    print('T-test for success picks')
    raggs = [x for x in raggs if math.isnan(x) == False]
    paggs = [x for x in paggs if math.isnan(x) == False]

    success_val_set = raggs
    success_train_set = paggs

    print('type: ', type(success_val_set))

    s_su, p_su = stats.ttest_ind(raggs, paggs)
    print(s_su, p_su)

    av = np.mean([p_fa,p_su])
    print('Average is: ', av)
    av_list.append(av)

    if av < av_ref:
        av_ref = av
        it_ref = it

    print("\n**** Real Statistics ****")
    print('          Mean,  Std, Cv')
    # print("Peak : ", statistics(rpeak_values))
    # print("Slope: ", statistics(rslopes))
    # print("AUCS : ", statistics(raucs))
    print("Feature : ", statistics(raggs))
    #
    print("\n**** Proxy Statistics ****")
    print('          Mean,  Std, Cv')
    # print("Peak : ", statistics(ppeak_values))
    # print("Slope: ", statistics(pslopes))
    # print("AUCS : ", statistics(paucs))
    print("Feature : ", statistics(paggs))

    # ... Plots ...
    df3 = pd.concat([real_success_df, proxy_success_df], axis=1, keys=['Real Tree', 'Apple Proxy']).stack(0)
    df3 = df3.reset_index(level=1)
    strip_and_box(df3, 'Agg Linear Trend', case, variable)

    # strip_and_box(df3, 'Peak [N]', case, variable)
    # strip_and_box(df3, 'AUC [N.s]', case, variable)

    # strip_and_box(df3, 'Slope [N/s]', case, variable)
    # count_plot(proxy_picks_shapes, real_picks_shapes, case, variable)

    plt.show()

    # ... Run classifier
    # Training Set
    X_train_f = failed_train_set
    y_train_f = [0]*len(X_train_f)

    X_train_s = success_train_set
    y_train_s = [1]*len(X_train_s)

    X_train = X_train_f + X_train_s
    y_train = y_train_f + y_train_s

    # Validation Set
    X_val_f = failed_val_set
    y_val_f = [0] * len(X_val_f)

    X_val_s = success_val_set
    y_val_s = [1] * len(X_val_s)

    X_val = X_val_f + X_val_s
    y_val = y_val_f + y_val_s

    print('Traini set')
    print(X_train)
    print(y_train)


    print('Validation set')
    print(X_val)
    print(y_val)



    print(av, it)
    print(av_list)

    rfc(X_train, y_train, X_val, y_val)

    # simple_classifier(X_train, y_train, X_val, y_val)