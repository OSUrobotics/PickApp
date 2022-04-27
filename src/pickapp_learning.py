# @Time : 3/16/2022 3:40 PM
# @Author : Alejandro Velasquez

"""
Sources:
https://machinelearningmastery.com/confusion-matrix-machine-learning/
"""

# Math related packages
import numpy as np
import os
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
# Visualization related packages
from tqdm import tqdm
import matplotlib.pyplot as plt
# Database related packages
import pandas as pd
import argparse


def rfc(experiments, depth, n_features):
    """
    Runs a Random Forest Classifier, to see if it can learn to differentiate successful picks from failed ones.
    It saves the report and plots into the results subfolder
    :param experiments: Number of Experiments to run, to overcome stochasticity of 'Random' Forest
    :param depth: Number of sub-branches that the classifier builds
    :param n_features: Number of feature to consider
    :return: none
    """

    print("Running RFC...")

    # --- features location
    # location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_4_for_tsfresh/'
    location = os.path.dirname(os.getcwd()) + '/data/features/'
    experiment = 'RFC with ' + 'TS-fresh features'

    # --- Train data
    train = 'best_features_TRAIN.csv'
    train_data = pd.read_csv(location + train)
    train_array = train_data.to_numpy()

    X_train = train_array[:, 1:(n_features + 1)]
    y_train = train_array[:, -1]

    # --- Test data
    test = 'best_features_TEST.csv'
    test_data = pd.read_csv(location + test)
    test_array = test_data.to_numpy()

    X_test = test_array[:, 1:(n_features + 1)]
    y_test = test_array[:, -1]

    # --- Scale Data
    # Makes all the data have a similar range
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    # Initialize variables
    data = []
    results = []
    max_acc = 0

    for i in tqdm(range(experiments)):

        # --- Train RF Classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=None)
        clf.fit(X_train, y_train)

        # --- Test Classifier
        performance = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        predictions = []

        for j, k in zip(X_test, y_test):
            grasp_prediction = clf.predict([j])
            predictions.append(grasp_prediction)

            if grasp_prediction == k:
                # Good Predictions
                performance += 1
                if grasp_prediction == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                # Bad Predictions
                if grasp_prediction == 1:
                    false_positives += 1
                else:
                    false_negatives += 1

        result = performance / len(X_test)

        # Only print the best Accuracy so far
        if result > max_acc:
            max_acc = result
            best_true_positives = true_positives
            best_false_positives = false_positives
            best_true_negatives = true_negatives
            best_false_negatives = false_negatives

            # Confusion Matrix from Scikit-Learn
            # Be aware that in this case the matrix columns are predictions, and rows are actual values
            # https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
            matrix = confusion_matrix(y_test, predictions, labels=[1, 0])
            report = classification_report(y_test, predictions, labels=[1, 0])
            # print("--- Scikit results ---")
            # print('Confusion Matrix: \n', matrix)
            # print('Classification Report: \n', report)

        # Append results for statistics
        results.append(result)

    # --- Confussion Matrix Report
    target_dir = os.path.dirname(os.getcwd()) + '/results/'
    name = 'ML_RFC__confussion_matrix.txt'

    with open(target_dir + name, 'w') as f:
        print("\nClassifier: Random Forest", file=f)
        print("Best Accuracy: %.2f" % max_acc, file=f)
        print("Confusion Matrix", file=f)
        print("                  Reference      ", file=f)
        print("Prediction    Success     Failure", file=f)
        print("   Success       %i          %i" % (best_true_positives, best_false_positives), file=f)
        print("   Failure       %i          %i" % (best_false_negatives, best_true_negatives), file=f)
        print('\n')

    # --- Boxplot of RFC results
    fig, ax = plt.subplots()
    ax.boxplot(results)
    plt.ylabel('Accuracy')
    plt.xlabel('Random Forest Depth')
    plt.grid()
    plt.title('%s + %i experiments)' % (experiment, experiments))
    plt.ylim([0, 1])
    # Save boxplot
    name = 'ML_RFC accuracy.png'
    target_dir = os.path.dirname(os.getcwd()) + '/results/'
    fig.savefig(target_dir + name)

    plt.show()


def mlpc(experiments, n_features):
    """
    Runs a Multi-Layer_Perceptron Classifier, to see if it can learn to differentiate successful picks from failed ones.
    It saves the report and plots into the results subfolder
    :param experiments:
    :param n_features:
    :return:
    """

    print("Running MLPC...")

    # --- Feature location
    location = 'C:/Users/15416/PycharmProjects/PickApp/data/Real Apples Data/improved data/grasp/Data_with_33_cols/postprocess_4_for_tsfresh/'
    experiment = 'MLPC with ' + 'TS-fresh features'

    # Train data
    train = 'best_features_TRAIN.csv'
    train_data = pd.read_csv(location + train)
    train_array = train_data.to_numpy()

    # Test data
    test = 'best_features_TEST.csv'
    test_data = pd.read_csv(location + test)
    test_array = test_data.to_numpy()

    X_train = train_array[:, 1:(n_features + 1)]
    y_train = train_array[:, -1]

    X_test = test_array[:, 1:(n_features + 1)]
    y_test = test_array[:, -1]

    # Scale the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    results = []
    max_acc = 0
    for i in tqdm(range(experiments)):

        clf = MLPClassifier(solver='adam', random_state=None, max_iter=3000, hidden_layer_sizes=50)
        # --- Train Classifier
        clf.fit(X_train, y_train)

        # --- Test Classifier
        performance = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for j, k in zip(X_test, y_test):
            grasp_prediction = clf.predict([j])
            # print(grasp_prediction)

            if grasp_prediction == k:
                # Good Predictions
                performance += 1

                if grasp_prediction == 1:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                # Bad Predictions
                if grasp_prediction == 1:
                    false_positives += 1
                else:
                    false_negatives += 1

        result = performance / len(X_test)

        # Only print the best Accuracy so far
        if result > max_acc:
            max_acc = result

            best_true_positives = true_positives
            best_false_positives = false_positives
            best_true_negatives = true_negatives
            best_false_negatives = false_negatives

        # Append results for statistics
        results.append(result)

    # --- Confussion Matrix Report
    target_dir = os.path.dirname(os.getcwd()) + '/results/'
    name = 'ML_MLPC__confussion_matrix.txt'

    with open(target_dir + name, 'w') as f:
        print("\nClassifier: Multi Layer Perceptron", file=f)
        print("Best Accuracy: %.2f" % max_acc, file=f)
        print("Confusion Matrix", file=f)
        print("                  Reference      ", file=f)
        print("Prediction    Success     Failure", file=f)
        print("   Success       %i          %i" % (best_true_positives, best_false_positives), file=f)
        print("   Failure       %i          %i" % (best_false_negatives, best_true_negatives), file=f)
        print('\n')

    fig, ax = plt.subplots()
    ax.boxplot(results)
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.grid()
    plt.title('%s + %i experiments )' % (experiment, experiments))
    plt.ylim([0, 1])

    # Save boxplot
    name = 'ML_MLPC accuracy.png'
    target_dir = os.path.dirname(os.getcwd()) + '/results/'
    fig.savefig(target_dir + name)

    plt.show()


def main():
    """
    This module runs machine learning classifiers.
    The user should give as arguments:
        a) machine learning classifier: Random Forest Classifier (RFC) or Multi Layer Perceptron Classifier (MLPC)
        b) number of experiments to run with each classifier
        c) number of features to consider from the data
        d) depth as a parameter for RFC
    :return:
    """

    # --- Parse Arguments from Command Line ---
    parser = argparse.ArgumentParser(description='This module runs machine learning classifiers. The user should give'
                                                 ' as arguments: '
                                                 ' a) machine learning classifier: Random Forest Classifier (RFC) or Multi Layer Perceptron Classifier (MLPC) '
                                                 ' b) number of experiments to run with each classifier '
                                                 ' c) number of features to consider from the data '
                                                 ' d) depth as a parameter for RFC')
    parser.add_argument('--experiments',
                        default=10,
                        type=int,
                        help='Number of experiments to run the classifier')
    parser.add_argument('--depth',
                        default=10,
                        type=int,
                        help='Depth of the random forest branches')
    parser.add_argument('--features',
                        default=1,
                        type=int,
                        help='Number of features to read')
    parser.add_argument('--classifier',
                        default='rfc',
                        type=str,
                        help='Machine Learning Classifier to implement: "rfc", "mlpc"')
    args = parser.parse_args()

    experiments = args.experiments
    depth = args.depth
    features = args.features

    if args.classifier == 'rfc':
        # Random Forest Classifier
        rfc(experiments, depth, features)
    elif args.classifier == 'mlpc':
        # Multi Layer Perceptron Classifier
        mlpc(experiments, features)


if __name__ == "__main__":
    main()