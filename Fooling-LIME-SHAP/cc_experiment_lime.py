"""
The LIME experiment MAIN for Communities and Crime.
 * Run the file and the CC experiments will complete
 * See compas experiment file for more details on how to read results.
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

from copy import deepcopy

import csv

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_cc(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

# Split the data in train and test set
data_train, data_test, ytrain, ytest = train_test_split(X, y, test_size=0.1)

# Save the data, so we can generate new samples in R
data_train["response"] = ytrain
data_train.to_csv("..\Data\cc_RBF_train.csv", index = False)

# Stops the execution of experiment so generators have time to generate data in R
input("Press enter, when rbfDataGen and treeEnsemble generated all the data.")

data_train = data_train.drop("response", axis = 1)
features = [c for c in data_train]

# Only categorical features are unrelated columns
categorical_features = ["unrelated_column_one", "unrelated_column_two"]
categorical_idcs = [features.index("unrelated_column_one"), features.index("unrelated_column_two")]
dummy_idcs = [[categorical_idcs[0]], [categorical_idcs[1]]]

# Find integer features
integer_attributes = [i for i, feature in enumerate(data_test.columns)
					if (data_test[feature].dtype in ["int64", "int32", "int8", "uint64", "uint32", "uint8"] and i not in categorical_idcs)]

# Parameters for MCD-VAE
latent_dim = data_train.shape[1] // 2
original_dim = data_train.shape[1]

# Indices of sensitive and unrelated features
race_indc = features.index('racePctWhite numeric')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

xtrain = data_train.values
xtest = data_test.values

# for racist model
mean_race = np.mean(xtrain[:,race_indc])

###
## The models f and psi for CC.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > mean_race else params.positive_outcome for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)
    
# the display model with one unrelated feature
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X): 
        return one_hot_encode(np.array([params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X]))

# the display model with two unrelated features
class innocuous_model_psi_two:
	def predict_proba(self, X):
		# Using 0.5 to make it easier to detect decision boundary on perturbation
		A = np.where(X[:,unrelated_indcs] > .5, params.positive_outcome, params.negative_outcome)
		B = np.where(X[:,unrelated_indcs1] < .5, params.positive_outcome, params.negative_outcome)
		preds = np.logical_xor(A, B).astype(int)
		return one_hot_encode(preds)
#
##
###

def experiment_main():
	"""
	Run through experiments for LIME on CC using both one and two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	print ('---------------------')
	print ("Beginning LIME CC Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
	print ('---------------------')

	# Dictionaries that will store adversarial models and explanation methods
	adv_models = dict()
	adv_explainers = dict()

	# Generator specifications
	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100,\
						"dropout": 0.3, "experiment": "CC"}

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
			file_name = f"../Rezultati/CCLime/CCLimeSummary_adversarial_{model}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])

	# Repeat the same thing for two features (innocuous_model_psi_two is used)
	adv_models = dict()
	adv_explainers = dict()

	# Generator specifications
	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.3, "experiment": "CC"}

	# Train the adversarial models for LIME with f and psi (fill te dictionary)
	adv_models["Perturbation"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain,\
								categorical_features=categorical_idcs, feature_names=features, perturbation_multiplier=1)
	adv_models["DropoutVAE"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two(),
								generator = "DropoutVAE", generator_specs = generator_specs).train(xtrain, ytrain,\
								categorical_features=categorical_idcs, integer_attributes = integer_attributes, feature_names=features, perturbation_multiplier=1)
	adv_models["RBF"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two(),\
								generator = "RBF", generator_specs = generator_specs).train(xtrain,\
								ytrain, feature_names=features, categorical_features=categorical_idcs)
	adv_models["Forest"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two(),\
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
			print (f"LIME Ranks and Pct Occurances (1 corresponds to most important feature) for two unrelated features\
			adversarial model: {model}, explainer: {explainer}:")
			summary = experiment_summary(explanations, features)
			print (summary)
			print ("Fidelity:", round(adv_lime.fidelity(xtest),2))

			# Save Resutls
			file_name = f"../Rezultati/CCLime/CCLimeSummary2_adversarial_{model}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])

if __name__ == "__main__":
	experiment_main()
