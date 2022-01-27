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


# Parameters
experiments = 10
maxdepth = 10


# Autoencoder Features (from pickle Files)
location = 'C:/Users/15416/Box/Learning to pick fruit/Apple Pick Data/Apple Proxy Picks/Winter 2022/grasp_classifer_data/'
subfolder = 'Autoencoders/Two Features/'

experiment = 'RFC with 2 Autoencoder features'


pck_X_train = location + subfolder + 'Autoencoder 2' + '.pickle'
X_train = pd.read_pickle(pck_X_train)
