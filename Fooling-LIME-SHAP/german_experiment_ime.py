"""
The IME experiment MAIN for GERMAN.
"""
import warnings
warnings.filterwarnings('ignore') 

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from sklearn.cluster import KMeans 

from copy import deepcopy

import csv

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")

# Prepare data (use this option if you are not using treeEnsemble)
'''
X, y, cols = get_and_preprocess_german(params)

# Merge the categorical features that represents the same attribute (bur are not one-hot encoded) into one feature
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

# Split the data into train and test set
data_train, data_test, ytrain, ytest = train_test_split(X, y, test_size=0.1)'''

# If data was split before the experiment (required for treeEnsemble)
data_train = pd.read_csv("..\Data\german_forest_train.csv")
data_test = pd.read_csv("..\Data\german_forest_test.csv")
ytrain = data_train.pop("response")
ytest = data_test.pop("response")

# MCD-VAE latent dim
latent_dim = data_train.shape[1] // 2

# Convert categorical features to one-hot encoded vector (this must be done after saving the data
# as rbfDataGen and treeEnsemble work with original values)
data_train = pd.get_dummies(data_train)
features = [c for c in data_train]

# Find categorical features
categorical_features = []
for feature in data_train.columns:
	if np.all(np.isin(data_train[feature], [0, 1])):
		categorical_features.append(feature)
print(categorical_features)
categorical_idcs = [i for (i, feature) in enumerate(data_train.columns) if feature in categorical_features]
# Indices of dummy variables
dummy_idcs = [[features.index('CheckingAccountBalance_geq_200'), features.index('CheckingAccountBalance_geq_0_lt_200'), features.index('CheckingAccountBalance_lt_0')], \
			[features.index('SavingsAccountBalance_geq_500'), features.index('SavingsAccountBalance_geq_100_lt_500'), features.index('SavingsAccountBalance_lt_100')], \
			[features.index('YearsAtCurrentJob_geq_4'), features.index('YearsAtCurrentJob_geq_1_lt_4'), features.index('YearsAtCurrentJob_lt_1')]]
print(dummy_idcs)	

data_test = pd.get_dummies(data_test)

# Check if all dummy variables are in test set, add the missing ones
for dummy in dummy_idcs:
	for idx in dummy:
		if features[idx] not in data_test.columns:
			data_test[features[idx]] = [0]*data_test.shape[0]
data_test = data_test[features]

print(list(data_test.columns))

# So user can check if dummy indices were correctly assigned
print(list(data_test.columns))

input("Check the dummy indices. Press enter to coninue.")

# Find integer features
integer_attributes = [i for i, feature in enumerate(data_test.columns)
					if (data_test[feature].dtype in ["int64", "int32", "int8", "uint64", "uint32", "uint8"] and i not in categorical_idcs)]

# Indices of sensitive and unrelated feature
gender_indc = features.index('Gender')
loan_rate_indc = features.index('LoanRateAsPercentOfIncome')

xtrain = data_train.values
xtest = data_test.values

original_dim = xtrain.shape[1]

mean_lrpi = np.mean(xtrain[:,loan_rate_indc])

# the biased model 
class racist_model_f:
    # Decision rule: classify negative outcome if female
    def predict(self,X):
        return np.array([params.negative_outcome if x[gender_indc] == 0 else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to loan rate indc
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[loan_rate_indc] > mean_lrpi else params.positive_outcome for x in X]))

##
###

def experiment_main():
	"""
	Run through experiments for IME on German,
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.2,\
						"experiment": "German", "feature_names": features}

	print ('---------------------')
	print ('Training adversarial models....')
	print ('---------------------')

	# Adversarial models
	adv_models = dict()
	adv_models["Perturbation"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain,\
														feature_names=features, perturbation_multiplier=1)
	adv_models["DropoutVAE"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi(), generator = "DropoutVAE", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_idcs, integer_idcs=integer_attributes, perturbation_multiplier=1)
	adv_models["ForestFill"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi(), generator = "Forest", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_idcs, integer_idcs=integer_attributes, perturbation_multiplier=1)

	for adversarial in ["Perturbation", "DropoutVAE", "ForestFill"]:
		adv_model = adv_models[adversarial]

		print ('---------------------')
		print (f'Training explainers with adversarial {adversarial}....')
		print ('---------------------')

		# Explainers
		adv_kernel_explainers = dict()
		adv_kernel_explainers["Perturbation"] = shap.SamplingExplainer(adv_model.predict, xtrain)
		adv_kernel_explainers["DropoutVAE"] = shap.SamplingExplainer(adv_model.predict, xtrain, generator="DropoutVAE", generator_specs=generator_specs,\
								dummy_idcs=dummy_idcs, integer_idcs=integer_attributes, instance_multiplier = 1000)
		adv_kernel_explainers["ForestFill"] = shap.SamplingExplainer(adv_model.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
								dummy_idcs=dummy_idcs, integer_idcs=integer_attributes)

		for explainer in ["Perturbation", "DropoutVAE", "ForestFill"]:
			adv_kernel_explainer = adv_kernel_explainers[explainer]
			explanations = adv_kernel_explainer.shap_values(xtest, fill_data=True, data_location="...\Data/german_forest_ime.csv", distribution_size=1000)

			# format for display
			formatted_explanations = []
			for exp in explanations:
				formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

			print (f"IME Ranks and Pct Occurances one unrelated feature, adversarial: {adversarial}, explainer: {explainer}:")
			summary = experiment_summary(formatted_explanations, features)
			print (summary)
			print ("Fidelity:",round(adv_model.fidelity(xtest),2))

			file_name = f"../Results/GermanIme/germanImeSummary_adversarial_{adversarial}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])

if __name__ == "__main__":
	experiment_main()
