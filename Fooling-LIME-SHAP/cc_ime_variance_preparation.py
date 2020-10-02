"""
Preparation for IME variance experiment on CC. This code splits the data into training and evaluation set,
trains the classifiers and calculates "true" Shapley values for each evaluation set instance for each classifier
"""
import warnings
warnings.filterwarnings('ignore') 

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from copy import deepcopy

import csv

import pickle

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_cc(params)

# add unrelated columns,
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

X["y"] = y
X = pd.get_dummies(X)

# Split the data into train and evaluation set
data_train, data_test = train_test_split(X, test_size=0.1)

data_train.to_csv("..\Data\IME\cc_train.csv", index = False)
data_test.to_csv("..\Data\IME\cc_test.csv", index = False)

ytrain = data_train.pop("y")
ytest = data_test.pop("y")
xtrain = data_train.values
xtest = data_test.values


"""
Naive Bayes
"""
print ('---------------------')
print ('Beginning with Naive Bayes')
print ('---------------------')

# Train naive bayes on train data
naive_bayes = GaussianNB()
naive_bayes.fit(X = xtrain, y = ytrain)

# We take for true Shapley values the ones we get with sampling with low error and probability of error
bayes_explainer = shap.SamplingExplainer(naive_bayes.predict, xtrain)
real_values_bayes = bayes_explainer.shap_values(xtest, nsamples="variance", alpha=0.99, expected_error=0.001)

# Save the classifier and shapley values
np.save("../Data/IME/CC/bayes_values", real_values_bayes)
filename = "../Data/IME/CC/bayes_model.sav"
pickle.dump(naive_bayes, open(filename, "wb"))

"""
Linear SVM
"""
print ('---------------------')
print ('Beginning with Linear SVM')
print ('---------------------')

# Train Linear SVM for classification on train data
lin_svm = LinearSVC()
lin_svm.fit(X = xtrain, y = ytrain)

# We take for true Shapley values the ones we get with sampling with low error and probability of error
svm_explainer = shap.SamplingExplainer(lin_svm.predict, xtrain)
real_values_svm = svm_explainer.shap_values(xtest, nsamples="variance", alpha=0.99, expected_error=0.001)

# Save the classifier and shapley values
np.save("../Data/IME/CC/svm_values", real_values_svm)
filename = "../Data/IME/CC/svm_model.sav"
pickle.dump(lin_svm, open(filename, "wb"))

"""
Random Forest
"""

print ('---------------------')
print ('Beginning with Random Forest')
print ('---------------------')

# Train Random Forest for classification on train data
forest = RandomForestClassifier()
forest.fit(X = xtrain, y = ytrain)

# We take for true Shapley values the ones we get with sampling with low error and probability of error
forest_explainer = shap.SamplingExplainer(forest.predict, xtrain)
real_values_forest = forest_explainer.shap_values(xtest, nsamples="variance", alpha=0.99, expected_error=0.001)

# Save the classifier and shapley values
np.save("../Data/IME/CC/forest_values", real_values_forest)
filename = "../Data/IME/CC/forest_model.sav"
pickle.dump(forest, open(filename, "wb"))

"""
Neural Network
"""
print ('---------------------')
print ('Beginning with Neural Network')
print ('---------------------')

# Shape of the NN
hidden_layers_size = (xtrain.shape[1] // 2, ((xtrain.shape[1] // 2) + 1) // 2)

# Train the NN on train data
network = MLPClassifier(hidden_layer_sizes=hidden_layers_size)
network.fit(X = xtrain, y = ytrain)

# We take for true Shapley values the ones we get with sampling with low error and probability of error
network_explainer = shap.SamplingExplainer(network.predict, xtrain)
real_values_nn = network_explainer.shap_values(xtest, nsamples="variance", alpha=0.99, expected_error=0.001)

# Save the classifier and shapley values
np.save("../Data/IME/CC/nn_values", real_values_nn)
filename = "../Data/IME/CC/nn_model.sav"
pickle.dump(network, open(filename, "wb"))