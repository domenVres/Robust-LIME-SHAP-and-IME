"""
Code that splits German data into train and evaluation set and saves it so it can be used in any German experiment for
treeEnsemble with data fill
"""

import numpy as np
import pandas as pd

from utils import *
from get_data import *

from sklearn.model_selection import train_test_split

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
X, y, cols = get_and_preprocess_german(params)

# Change one-hot-encoded features to normal categorical features
geq0 = X['CheckingAccountBalance_geq_0']
geq200 = X['CheckingAccountBalance_geq_200']
checkingAccountBalance = ["geq_200" if geq200[i] == 1 else ("geq_0_lt_200" if geq0[i] == 1 else "lt_0") for i in range(X.shape[0])]
X["CheckingAccountBalance"] = checkingAccountBalance
X = X.drop(labels=['CheckingAccountBalance_geq_0', 'CheckingAccountBalance_geq_200'], axis = 1)

geq100 = X['SavingsAccountBalance_geq_100']
geq500 = X['SavingsAccountBalance_geq_500']
savingsAccountBalance = ["geq_500" if geq500[i] == 1 else ("geq_100_lt_500" if geq100[i] == 1 else "lt_100") for i in range(X.shape[0])]
X["SavingsAccountBalance"] = savingsAccountBalance
X = X.drop(labels=['SavingsAccountBalance_geq_100', 'SavingsAccountBalance_geq_500'], axis = 1)

lt1 = X['YearsAtCurrentJob_lt_1']
geq4 = X['YearsAtCurrentJob_geq_4']
yearsAtCurrentJob = ["geq_4" if geq4[i] == 1 else ("geq_1_lt_4" if lt1[i] == 0 else "lt_1") for i in range(X.shape[0])]
X["YearsAtCurrentJob"] = yearsAtCurrentJob
X = X.drop(labels = ['YearsAtCurrentJob_lt_1', 'YearsAtCurrentJob_geq_4'], axis = 1)

X["response"] = y

# Split data into train and test set
data_train, data_test, ytrain, ytest = train_test_split(X, y, test_size=0.1)

# Save the data so we can generate it in R
data_train.to_csv("..\Data\german_forest_train.csv", index = False)
data_test.to_csv("..\Data\german_forest_test.csv", index = False)