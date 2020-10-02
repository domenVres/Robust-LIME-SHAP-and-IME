"""
Code that splits CC data into train and evaluation set and saves it so it can be used in any CC experiment for
treeEnsemble with data fill
"""

import numpy as np
import pandas as pd

from utils import *
from get_data import *

from sklearn.model_selection import train_test_split

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_cc(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

X["response"] = y

# Split the data into train and test set
data_train, data_test, ytrain, ytest = train_test_split(X, y, test_size=0.1)

# Save the data so we can generate it in R
data_train.to_csv("..\Data\cc_forest_train.csv", index = False)
data_test.to_csv("..\Data\cc_forest_test.csv", index = False)