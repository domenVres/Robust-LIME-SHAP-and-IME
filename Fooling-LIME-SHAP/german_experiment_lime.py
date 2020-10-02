"""
The LIME experiment MAIN for GERMAN.
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

from sklearn.cluster import KMeans 

from copy import deepcopy

from Generators.DropoutVAE import DropoutVAE

import csv

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
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
data_train, data_test, ytrain, ytest = train_test_split(X, y, test_size=0.1)

# Save the data, so we can generate new samples in R
data_train["response"] = ytrain
data_train.to_csv("..\Data\german_RBF_train.csv", index = False)

# Stops the execution of experiment so generators have time to generate data in R
input("Press enter, when rbfDataGen and treeEnsemble generated all the data.")

# Response was needed only by rbfDataGen and treeEnsemble
data_train = data_train.drop("response", axis = 1)
latent_dim = data_train.shape[1] // 2

# Convert categorical features to one-hot encoded vector (this must be done after saving the data
# as rbfDataGen and treeEnsemble work with original values)
data_test = pd.get_dummies(data_test)
data_train = pd.get_dummies(data_train)
features = [c for c in data_train]

# Find categorical features
categorical_features = []
for feature in data_train.columns:
	if np.all(np.isin(data_train[feature], [0, 1])):
		categorical_features.append(feature)
print(categorical_features)
categorical_idcs = [i for (i, feature) in enumerate(data_train.columns) if feature in categorical_features]
# Save the dummy indices
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

###
## The models f and psi for GERMAN.  We discriminate based on gender for f and consider loan rate % income for explanation
#

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
	Run through experiments for LIME on GERMAN.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	print ('---------------------')
	print ("Beginning LIME GERMAN Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
	print ('---------------------')
	

	# Dictionaries that will store adversarial models and explanation methods
	adv_models = dict()
	adv_explainers = dict()

	# Generator specifications
	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100,\
					"dropout": 0.3, "experiment": "German"}

	# Train the adversarial models for LIME with f and psi (fill te dictionary)
	adv_models["Perturbation"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain,\
								categorical_features=categorical_idcs, feature_names=features, perturbation_multiplier=1)
	adv_models["DropoutVAE"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi(),
								generator = "DropoutVAE", generator_specs = generator_specs).train(xtrain, ytrain,\
								categorical_features=categorical_idcs, integer_attributes = integer_attributes, feature_names=features, perturbation_multiplier=1)
	adv_models["RBF"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi(),\
								generator = "RBF", generator_specs = generator_specs).train(xtrain,\
								ytrain, feature_names=features, categorical_features=categorical_idcs)
	adv_models["Forest"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi(),\
								generator = "Forest", generator_specs = generator_specs).train(xtrain,\
								ytrain, feature_names=features, categorical_features=categorical_idcs)

	# Fill the dictionary with explanation methods
	for generator in ["Perturbation", "DropoutVAE", "RBF", "Forest"]:
		adv_explainers[generator] = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_models[generator].get_column_names(),\
									discretize_continuous=False, categorical_features=categorical_idcs, generator=generator,\
									generator_specs=generator_specs, dummies=dummy_idcs, integer_attributes=integer_attributes)

	# We check every combination of adversarial model/explanation method
	for model in adv_models:
		for explainer in adv_explainers:
			adv_lime = adv_models[model]
			adv_explainer = adv_explainers[explainer]
			explanations = []
			for i in range(xtest.shape[0]):
				explanations.append(adv_explainer.explain_instance(xtest[i], adv_lime.predict_proba).as_list())

			# Display Results
			print (f"LIME Ranks and Pct Occurances (1 corresponds to most important feature) for one unrelated feature\
			adversarial model: {model}, explainer: {explainer}:")
			summary = experiment_summary(explanations, features)
			print (summary)
			print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

			# Save Resutls
			file_name = f"../Results/GermanLime/germanLimeSummary_adversarial_{model}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])

if __name__ == "__main__":
	experiment_main()
