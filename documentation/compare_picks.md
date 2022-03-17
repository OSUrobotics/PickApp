Module compare_picks
====================

Functions
---------

    
`agg_linear_trend(x)`
:   Source: https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#agg_linear_trend
    
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

    
`compare_picks(reals, proxys, main, datasets, subfolder, case, variable, phase)`
:   Compares the apple picks element-wise from the reals and proxys lists
    :param phase: when the dynamic time warping is going to take place
    :param reals: list of picks from real tree
    :param proxys: list of picks from proxy
    :param main: main folder location
    :param datasets: list of real and proxy datasets
    :param subfolder: subfolder location
    :param case: whether successful or failed picks
    :param variable: channel of interest
    :return: none

    
`crossings(x, y)`
:   Checks the initial and ending time of the Force Profile based on the zero crossings
    :param x: time
    :param y: values
    :return: initial and ending time and its respective indexes

    
`main()`
:   

    
`number_from_filename(filename)`
:   Subtracts the number of an apple pick from its filename
    :param filename:
    :return: pick number

    
`pic_list(file, variable)`
:   Reads a csv file and returns only tha variable of interest and Time.
    This is useful because each topic in ROS saves several channels in one single csv file. Hence we want get only the
    data of the channel that we are interested in.
    :param file: csv file
    :param variable: Given as a string
    :return: Simplified lists (Time list, and values list)

    
`pick_info_from_metadata(location, file, index)`
:   Extracts the info from a certain index in the metadata file
    :param location: location of the metadata file
    :param file: metadata file
    :param index: index / column where we want to obtain the information from
    :return: information

    
`pick_subplot(axrray, phase, real_times, real_values, proxy_times, proxy_values, variable)`
:   Creat the subplots of the 'Grasp' and 'Pick' phase of an aple pick
    :param axrray: array of subplots
    :param phase: whether 'Grasp' or 'Pick'
    :param real_times: python list with the time values from real pick
    :param real_values: python list with the variable values from real pick
    :param proxy_times: python list with the time values from proxy pick
    :param proxy_values: python list with the variable values from proxy pick
    :param variable: channel of interest
    :return: none

    
`same_pose_lowest_noise_picks(real_picks_location, proxy_picks_location, label)`
:   Compares the labels from real and proxy picks and outputs lists of pairs with the same pose, same label and lowest
    noise (which represents the closest proxy pick)
    :param real_picks_location: folder with real apple picks
    :param proxy_picks_location: folder with proxy picks
    :param label: whether success or failed picks
    :return: Real and Pick list, where the picks are comparable element-wise

    
`same_pose_picks(real_picks_location, proxy_picks_location, label)`
:   Compares the labels from real and proxy picks and outputs lists of pairs with the same pose and label
    :param real_picks_location: folder with real apple picks
    :param proxy_picks_location: folder with proxy picks
    :param label: whether success or failed picks
    :return: Real and Pick list, where the picks are comparable element-wise

    
`topic_from_variable(variable)`
:   Given a variable, it returns the ROS topic associated to it
    :param variable:
    :return: topic