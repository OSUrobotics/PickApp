Module machine_learning
=======================
Sources:
https://machinelearningmastery.com/confusion-matrix-machine-learning/

Functions
---------

    
`main()`
:   This module runs machine learning classifiers.
    The user should give as arguments:
        a) machine learning classifier: Random Forest Classifier (RFC) or Multi Layer Perceptron Classifier (MLPC)
        b) number of experiments to run with each classifier
        c) number of features to consider from the data
        d) depth as a parameter for RFC
    :return:

    
`mlpc(experiments, n_features)`
:   Runs a Multi-Layer_Perceptron Classifier, to see if it can learn to differentiate successful picks from failed ones.
    It saves the report and plots into the results subfolder
    :param experiments:
    :param n_features:
    :return:

    
`rfc(experiments, depth, n_features)`
:   Runs a Random Forest Classifier, to see if it can learn to differentiate successful picks from failed ones.
    It saves the report and plots into the results subfolder
    :param experiments: Number of Experiments to run, to overcome stochasticity of 'Random' Forest
    :param depth: Number of sub-branches that the classifier builds
    :param n_features: Number of feature to consider
    :return: none