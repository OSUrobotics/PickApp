"""
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
# Visualization related packages
from tqdm import tqdm
import matplotlib.pyplot as plt
# Database related packages
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Parameters
n_features = 16
experiments = 20


Features = [2, 4, 8, 16, 33]
Features = [16]

location = 'Features/Experiment 3x1/'
experiment = 'features_exp3_by1_'


# ------------------------------------- Step 1: Load the Features --------------------------------------------------
pck_X_train = location + experiment + str(n_features) + '_train.pickle'
X_train = pd.read_pickle(pck_X_train)
pck_X_test = location + experiment + str(n_features) + '_test.pickle'
X_test = pd.read_pickle(pck_X_test)

pck_y_train = location + 'outputs_exp3_by1_' + 'train.pickle'
y_train = pd.read_pickle(pck_y_train)
pck_y_test = location + 'outputs_exp3_by1_' + 'test.pickle'
y_test = pd.read_pickle(pck_y_test)

scaler = MinMaxScaler()
scaler.fit(X_train)

# Scale the dataa
X_train = scaler.transform(X_train)

plt.plot(X_train)
plt.show()




#     results = []
#     for i in tqdm(range(experiments)):
#         # ---------------------------------- Step 3: Train and Test the classifier -------------------------------------
#         # Train Random Forest Classifier
#         # clf = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=None)
#         clf = MLPClassifier(solver='adam', random_state=None, max_iter=1000)
#         # clf = NearestCentroid()
#         # clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
#         clf.fit(X_train, y_train)
#
#         # Test it!
#         performance = 0
#         true_positives = 0
#         false_positives = 0
#         false_negatives = 0
#         true_negatives = 0
#         for j, k in zip(X_test, y_test):
#             grasp_prediction = clf.predict([j])
#             # print(grasp_prediction)
#
#             if grasp_prediction == k:
#                 # print("yeahh")
#                 performance += 1
#
#                 if grasp_prediction == 1:
#                     true_positives += 1
#                 else:
#                     true_negatives += 1
#             else:
#                 if grasp_prediction == 1:
#                     false_positives += 1
#                 else:
#                     false_negatives += 1
#
#         result = performance / len(X_test)
#         print("\nClassifier: Random Forest")
#         print("Parameters: %i tsfresh features" % n_features)
#         print("Accuracy: %.2f" % result)
#         print("Confusion Matrix")
#         print("                  Reference      ")
#         print("Prediction    Success     Failure")
#         print("   Success       %i          %i" % (true_positives, false_positives))
#         print("   Failure       %i          %i" % (false_negatives, true_negatives))
#         print('\n')
#
#         # Append results for statistics
#         results.append(result)
#
#     # print(results)
#     mean = np.mean(results)
#     st_dev = np.std(results)
#
#     data.append(results)
#
# fig, ax = plt.subplots()
# ax.boxplot(data)
# plt.ylabel('Accuracy')
# plt.xlabel('Number of Features')
# plt.grid()
# plt.title('RFC + Autoencoder + %s (%i experiments, %i features)' % (experiment, experiments, n_features))
# plt.ylim([0, 1])
# plt.show()
#
#
# # print("Random Forest accuracy (%i top features, %i experiments) = %.2f with std dev %.2f: " % (n_features, experiments, mean, st_dev))
#
# print("Random Forest accuracy (%i autoencoder features, %i experiments) = %.2f with std dev %.2f: " % (n_features, experiments, mean, st_dev))
#
#
