Module metadata_stats
=====================
This file opens all the csvs that have the labels from the experiments, and returns some basic statistics

Functions
---------

    
`main()`
:   This modules brings basic statistics of the metadata files from all the experiments.
    It outputs the % for each label and the distribution of the cartesian and angular noise that was applied
    during the picks performed at the apple proxy.
    :return:

Classes
-------

`MetadataStats()`
:   

    ### Methods

    `get_info(self, column)`
    :   Extract values at the column of each metadata file, and concatenate it into a single list

    `label_counter(self)`
    :   Counts the success and failures from the metadata files of a single dataset

    `noise_stats(self, data)`
    :   Outputs mean, st dev and percentiles of the noise of all the experiments form the data sets.
        It saves the report and boxplots in the results folder.
        :param data:
        :return: