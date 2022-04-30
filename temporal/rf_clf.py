"""

Sources:
https://machinelearningmastery.com/confusion-matrix-machine-learning/
"""

# Math related packages
import numpy as np
from numpy import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
# Visualization related packages
from tqdm import tqdm
import matplotlib.pyplot as plt
# Database related packages
import pandas as pd


# Parameters
n_features = 1
experiments = 20
sets_ratio = 0.30
depth = 10


# ------------------------------------- Step 1: Load the Features ------------------------------------------------------
# Tsfresh Features (from csv files)
location = '/home/avl/PycharmProjects/AppleProxy/Features/proxy dataset/'
name = '5x_data_joined.csv'

trial = pd.read_csv(location + name)
trial_array = trial.to_numpy()

# Autoencoder Features (from pickle Files)
# location_2 = 'Features/Experiment 1x1/'
# test_input = 'features_exp1_by1_test.pickle'
# auto_test_in = pd.read_pickle(location_2 + test_input)
#
# test_output = 'outputs_5byte.pickle'
# auto_test_out = pd.read_pickle(location_2 + test_output)


# Inputs
# Get only the most important n features
X = trial_array[:, 1:(n_features + 1)]
# X = auto_test_in

# Outputs
y = trial_array[:, -1]      # Last column on the right
# y = auto_test_out

X_success = []
y_success = []
X_failed = []
y_failed = []

# Split the data in two groups: success and failures, in order to control the ratio in train and test sets
for r, s in zip(X, y):
    if s == 1:
        X_success.append(r)
    else:
        X_failed.append(r)

y_success = [1] * len(X_success)
y_failed = [0] * len(X_failed)


# -------------------------------- Step 2: Split the data into Train and Test datasets ---------------------------------

data = []
for depth in range(1, depth + 1):

    results = []
    for i in tqdm(range(experiments)):
        X_train = []
        X_test = []
        y_train = []
        y_test = []

        X_train_success = []
        X_test_success = []
        y_train_success = []
        y_test_success = []
        X_train_failed = []
        X_test_failed = []
        y_train_failed = []
        y_test_failed = []

        # Perform 1 experiment, by choosing randomly the training set and test set
        # Pick data from the success dataset
        for i in range(len(X_success)):
            p = random.rand()

            if p < sets_ratio:
                # Only 30% of the data goes into the Testing Set
                X_test_success.append(X_success[i])
                y_test_success.append(y_success[i])
            else:
                # The rest (1-p)% goes into the Training Set
                X_train_success.append(X_success[i])
                y_train_success.append(y_success[i])

        # Pick data from the failed dataset
        for j in range(len(X_failed)):
            p = random.rand()

            if p < sets_ratio:
                # Only a % of the data goes into the Testing Set
                X_test_failed.append(X_failed[j])
                y_test_failed.append(y_failed[j])
            else:
                # The rest (1-p)% goes into the Training Set
                X_train_failed.append(X_failed[j])
                y_train_failed.append(y_failed[j])

        X_train = np.concatenate((X_train_failed, X_train_success), axis=0)
        y_train = np.concatenate((y_train_failed, y_train_success), axis=0)

        X_test = np.concatenate((X_test_failed, X_test_success), axis=0)
        y_test = np.concatenate((y_test_failed, y_test_success), axis=0)

        # ---------------------------------- Step 3: Train and Test the classifier -----------------------------------------
        # Train Random Forest Classifier
        clf = RandomForestClassifier(max_depth=depth, random_state=0)
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
        # print("\nClassifier: Random Forest")
        # print("Parameters: %i tsfresh features" % n_features)
        # print("Accuracy: %.2f" % result)
        # print("Confusion Matrix")
        # print("                  Reference      ")
        # print("Prediction    Success     Failure")
        # print("   Success       %i          %i" % (true_positives, false_positives))
        # print("   Failure       %i          %i" % (false_negatives, true_negatives))
        # print('\n')

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
plt.title('TAutoencoder with data x 5 (%i experiments, %i features)' % (experiments, n_features))
plt.ylim([0,1])
plt.show()

# print("Random Forest accuracy (%i top features, %i experiments) = %.2f with std dev %.2f: " % (n_features, experiments, mean, st_dev))

print("Random Forest accuracy (%i autoencoder features, %i experiments) = %.2f with std dev %.2f: " % (n_features, experiments, mean, st_dev))


