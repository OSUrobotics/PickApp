"""
Sources:
https://machinelearningmastery.com/confusion-matrix-machine-learning/
"""
# Math related packages
import numpy as np
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Visualization related packages
from tqdm import tqdm
import matplotlib.pyplot as plt
# Database related packages
import pandas as pd


data = []



# Parameters
n_features = 33
experiments = 10
depth = 3

# Autoencoder Features (from pickle Files)
location = 'Features/Experiment 1x5/'
experiment = 'exp1_by5_'


pck_X_train = location + 'features_' + experiment + str(n_features) + '_train.pickle'
X_train = pd.read_pickle(pck_X_train)
pck_X_test = location + 'features_' + experiment + str(n_features) + '_test.pickle'
X_test = pd.read_pickle(pck_X_test)

pck_y_train = location + 'outputs_' + experiment + 'train.pickle'
y_train = pd.read_pickle(pck_y_train)
pck_y_test = location + 'outputs_' + experiment + 'test.pickle'
y_test = pd.read_pickle(pck_y_test)

# Scale the data
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)


for depth in range(depth, depth + 1):

    results = []
    for i in tqdm(range(experiments)):
        # ---------------------------------- Step 3: Train and Test the classifier -------------------------------------
        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=250, max_depth=depth, random_state=None)
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
        experiment = 'real x5'
        label_1 = 'Experiment 1\n(' + str(experiment) + ',\n ' + str(n_features) + 'features' + ', ' + str(depth) + 'depth)'

        # Append results for statistics
        results.append(result)

    # print(results)
    mean = np.mean(results)
    st_dev = np.std(results)

    data.append(results)




# Parameters
n_features = 2
experiments = 10
depth = 7

# Autoencoder Features (from pickle Files)
location = 'Features/Experiment 2x5/'
experiment = 'exp2_by5_'

pck_X_train = location + 'features_' + experiment + str(n_features) + '_train.pickle'
X_train = pd.read_pickle(pck_X_train)
pck_X_test = location + 'features_' + experiment + str(n_features) + '_test.pickle'
X_test = pd.read_pickle(pck_X_test)

pck_y_train = location + 'outputs_' + experiment + 'train.pickle'
y_train = pd.read_pickle(pck_y_train)
pck_y_test = location + 'outputs_' + experiment + 'test.pickle'
y_test = pd.read_pickle(pck_y_test)

# Scale the data
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)


for depth in range(depth, depth + 1):

    results = []
    for i in tqdm(range(experiments)):
        # ---------------------------------- Step 3: Train and Test the classifier -------------------------------------
        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=250, max_depth=depth, random_state=None)
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
        experiment = 'proxy train & real test x5'
        label_2 = 'Experiment 2\n(' + str(experiment) + ',\n ' + str(n_features) + 'features' + ', ' + str(depth) + 'depth)'

        # Append results for statistics
        results.append(result)

    # print(results)
    mean = np.mean(results)
    st_dev = np.std(results)

    data.append(results)




# Parameters
n_features = 2
experiments = 10
depth = 8

# Autoencoder Features (from pickle Files)
location = 'Features/Experiment 3x5/'
experiment = 'exp3_by5_'

pck_X_train = location + 'features_' + experiment + str(n_features) + '_train.pickle'
X_train = pd.read_pickle(pck_X_train)
pck_X_test = location + 'features_' + experiment + str(n_features) + '_test.pickle'
X_test = pd.read_pickle(pck_X_test)

pck_y_train = location + 'outputs_' + experiment + 'train.pickle'
y_train = pd.read_pickle(pck_y_train)
pck_y_test = location + 'outputs_' + experiment + 'test.pickle'
y_test = pd.read_pickle(pck_y_test)

# Scale the data
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)


for depth in range(depth, depth + 1):

    results = []
    for i in tqdm(range(experiments)):
        # ---------------------------------- Step 3: Train and Test the classifier -------------------------------------
        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=250, max_depth=depth, random_state=None)
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
        experiment = 'proxy + real train & real test x5'
        label_3 = 'Experiment 3\n(' + str(experiment) + ',\n ' + str(n_features) + 'features' + ', ' + str(depth) + 'depth)'

        # Append results for statistics
        results.append(result)

    # print(results)
    mean = np.mean(results)
    st_dev = np.std(results)

    data.append(results)




fig, ax = plt.subplots()
ax.boxplot(data, widths=0.6)
plt.ylabel('Accuracy')
# plt.xlabel('Random Forest Depth')
plt.grid()
plt.title('RFC / Autoencoder features / %i trials' % (experiments))
plt.ylim([0.5, 1])
ax.set_xticklabels([label_1, label_2, label_3])
plt.show()
