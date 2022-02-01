"""
Multi Layer Perceptron Classifier - MLCP
Sources:
https://machinelearningmastery.com/confusion-matrix-machine-learning/
"""
# Math related packages
import numpy as np
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Visualization related packages
from tqdm import tqdm
import matplotlib.pyplot as plt
# Database related packages
import pandas as pd


# Parameters
experiments = 10
Features = [2, 4, 8, 16, 32]


data = []
for n_features in Features:
    # ------------------------------------- Step 1: Load the Features --------------------------------------------------
    # --- Autoencoder Features ---
    # Autoencoder Features (from pickle Files)
    location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/' \
               '/Apple Proxy Picks/Winter 2022/grasp_classifer_data/Autoencoders/'
    subfolder = 'Set 4/'
    features = n_features
    experiment = 'MLPC with ' + str(features) + 'Autoencoder features'

    pck_X_train = location + subfolder + 'Autoencoder Set 4 ' + str(features) + ' Training Inputs.pickle'
    X_train = pd.read_pickle(pck_X_train)
    pck_X_test = location + subfolder + 'Autoencoder Set 4 ' + str(features) + ' Testing Inputs.pickle'
    X_test = pd.read_pickle(pck_X_test)

    pck_y_train = location + subfolder + 'outputs_Set4_train.pickle'
    y_train = pd.read_pickle(pck_y_train)
    pck_y_test = location + subfolder + 'outputs_Set4_test.pickle'
    y_test = pd.read_pickle(pck_y_test)

    # # --- ts-fresh features ---
    # location = '/home/avl/PycharmProjects/AppleProxy/Features/real dataset/'
    # experiment = 'real dataset x5'
    #
    # # Train data
    # train = '5x_data_train.csv'
    # train_data = pd.read_csv(location + train)
    # train_array = train_data.to_numpy()
    #
    # # Test data
    # test = '5x_data_test.csv'
    # test_data = pd.read_csv(location + test)
    # test_array = test_data.to_numpy()
    #
    # X_train = train_array[:, 1:(n_features + 1)]
    # y_train = train_array[:, -1]
    #
    # X_test = test_array[:, 1:(n_features + 1)]
    # y_test = test_array[:, -1]

    # Scale the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    # scaler.fit(y_train)
    # y_train = scaler.transform(y_train)
    # scaler.fit(y_test)
    # y_test = scaler.transform(y_test)

    results = []
    max_acc = 0
    for i in tqdm(range(experiments)):
        # ---------------------------------- Step 3: Train and Test the classifier -------------------------------------
        # Train Random Forest Classifier
        # clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=None)
        clf = MLPClassifier(solver='adam', random_state=None, max_iter=10000, hidden_layer_sizes= 1000)
        # clf = NearestCentroid()
        # clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
        clf.fit(X_train, y_train)

        # Test it!
        performance = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for j, k in zip(X_test, y_test):
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

        # Only print the best Accuracy so far
        if result > max_acc:
            max_acc = result
            print("\nSo far the best has been:")
            print("\nClassifier: Multi Layer Perceptron")
            print("Parameters: %i Autoencoder features" % n_features)
            print("Accuracy: %.2f" % result)
            print("Confusion Matrix")
            print("                  Reference      ")
            print("Prediction    Success     Failure")
            print("   Success       %i          %i" % (true_positives, false_positives))
            print("   Failure       %i          %i" % (false_negatives, true_negatives))
            print('\n')

        # Append results for statistics
        results.append(result)

    # print(results)
    mean = np.mean(results)
    st_dev = np.std(results)

    data.append(results)

fig, ax = plt.subplots()
ax.boxplot(data)
plt.ylabel('Accuracy')
plt.xlabel('Number of Features')
plt.grid()
plt.title('%s + %i experiments )' % (experiment, experiments))
plt.ylim([0, 1])
plt.show()


