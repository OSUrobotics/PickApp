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
from sklearn.metrics import confusion_matrix, classification_report
# Visualization related packages
from tqdm import tqdm
import matplotlib.pyplot as plt
# Database related packages
import pandas as pd


# Parameters
experiments = 10
maxdepth = 10


# Autoencoder Features (from pickle Files)
location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/' \
            '/Apple Proxy Picks/Winter 2022/grasp_classifer_data/Autoencoders/'
subfolder = 'Set 1/'
features = 32

experiment = 'RFC with ' + str(features) + 'Autoencoder features'


pck_X_train = location + subfolder + 'Autoencoder Set 1 ' + str(features) + ' Training Inputs.pickle'
X_train = pd.read_pickle(pck_X_train)
pck_X_test = location + subfolder + 'Autoencoder Set 1 ' + str(features) + ' Testing Inputs.pickle'
X_test = pd.read_pickle(pck_X_test)

pck_y_train = location + subfolder + 'outputs_Set1_train.pickle'
y_train = pd.read_pickle(pck_y_train)
pck_y_test = location + subfolder + 'outputs_Set1_test.pickle'
y_test = pd.read_pickle(pck_y_test)

# print(y_train)

# Scale the data
scaler = MinMaxScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)


data = []
for depth in range(1, maxdepth + 1, 2):

    results = []
    max_acc = 0

    for i in tqdm(range(experiments)):
        # ---------------------------------- Step 3: Train and Test the classifier -------------------------------------
        # Train Random Forest Classifier
        clf = RandomForestClassifier(n_estimators=1000, max_depth=depth, random_state=None)
        clf.fit(X_train, y_train)

        # Test it!
        performance = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        predictions = []

        for j, k in zip(X_test, y_test):
            grasp_prediction = clf.predict([j])
            # print(grasp_prediction)

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
            print("\nSo far the best has been:")
            print("\nClassifier: Random Forest")
            # print("Parameters: %i tsfresh features" % n_features)
            print("Accuracy: %.2f" % result)
            print("Confusion Matrix")
            print("                  Reference      ")
            print("Prediction    Success     Failure")
            print("   Success       %i          %i" % (true_positives, false_positives))
            print("   Failure       %i          %i" % (false_negatives, true_negatives))
            print('\n')

        # Confusion Matrix from Scikit-Learn
        # Be aware that in this case the matrix columns are predictions, and rows are actual values
        # https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
        matrix = confusion_matrix(y_test, predictions, labels=[1, 0])
        report = classification_report(y_test, predictions, labels=[1, 0])
        print("--- Scikit results ---")
        print('Confusion Matrix: \n', matrix)
        print('Classification Report: \n', report)

        # Append results for statistics
        results.append(result)

    # print(results)
    mean = np.mean(results)
    st_dev = np.std(results)
    data.append(results)

fig, ax = plt.subplots()
ax.boxplot(data)
plt.ylabel('Accuracy')
plt.xlabel('Random Forest Depth')
plt.grid()
plt.title('%s + %i experiments)' % (experiment, experiments))
plt.ylim([0, 1])
plt.show()

