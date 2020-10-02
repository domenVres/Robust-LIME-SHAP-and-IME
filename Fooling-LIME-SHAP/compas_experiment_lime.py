"""
The LIME experiment MAIN for COMPAS.
 * Run the file and the COMPAS experiments will complete
 * This may take some time because we iterate through every instance in the test set for
   LIME explanations take some time to compute
 * The print outs can be interpreted as maps from the RANK to the rate at which the feature occurs in the rank.. e.g:
 	    1: [('length_of_stay', 0.002592352559948153), ('unrelated_column_one', 0.9974076474400518)]
   can be read as the first unrelated column occurs ~100% of the time in as the most important feature
 * "Nothing shown" refers to LIME yielding only 0 contributions 
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

from copy import deepcopy

import csv

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")
np.random.seed(params.seed)
X, y, cols = get_and_preprocess_compas_data_RBF(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

data_train, data_test, ytrain, ytest = train_test_split(X, y, test_size=0.1)

# Train data needs to be saved so it can be generated in R
data_train["response"] = ytrain
data_train.to_csv("..\Data\compas_RBF_train.csv", index = False)

# Stops the execution of experiment so generators have time to generate data in R
input("Press enter, when rbfDataGen and treeEnsemble generated all the data.")

# Convert categorical features to one-hot encoded vector (this must be done after saving the data
# as rbfDataGen and treeEnsemble work with original values)
data_test = pd.get_dummies(data_test)

# List of feature names and indices required by the explanation method
features = [c for c in data_test]
categorical_feature_name = ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M',\
                            'sex_Female', 'sex_Male', 'race', 'unrelated_column_one', 'unrelated_column_two']
categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]
dummy_indcs = [[categorical_feature_indcs[0]], [categorical_feature_indcs[1], categorical_feature_indcs[2]],\
            [categorical_feature_indcs[3], categorical_feature_indcs[4]], [categorical_feature_indcs[5]],\
			[categorical_feature_indcs[6]], [categorical_feature_indcs[7]]]

# Find all integer features
integer_attributes = [i for i, feature in enumerate(data_test.columns)
					if (data_test[feature].dtype in ["int64", "int32", "int8", "uint64", "uint32", "uint8"] and i not in categorical_feature_indcs)]

# Response was needed only by rbfDataGen and treeEnsemble
data_train = data_train.drop("response", axis = 1)
# MCD-VAE latent dimension
latent_dim = data_train.shape[1] // 2
data_train = pd.get_dummies(data_train)

# Indices of sensitive and unrelated features
race_indc = features.index('race')
unrelated_indcs = features.index('unrelated_column_one')
unrelated_indcs1 = features.index('unrelated_column_two')

xtrain = data_train.values
xtest = data_test.values

original_dim = xtrain.shape[1]

###
## The models f and psi for COMPAS.  We discriminate based on race for f and concider two RANDOMLY DRAWN features to display in psi
#

# the biased model 
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

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
		A = np.where(X[:,unrelated_indcs] > 0, params.positive_outcome, params.negative_outcome)
		B = np.where(X[:,unrelated_indcs1] > 0, params.positive_outcome, params.negative_outcome)
		preds = np.logical_xor(A, B).astype(int)
		return one_hot_encode(preds)
#
##
###

def experiment_main():
	"""
	Run through experiments (every combination adversarial/explainer) for LIME on compas using both one and two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	print ('---------------------')
	print ("Beginning LIME COMPAS Experiments....")
	print ("(These take some time to run because we have to generate explanations for every point in the test set) ")
	print ('---------------------')

	# Dictionaries that will store adversarial models and explanation methods
	adv_models = dict()
	adv_explainers = dict()

	# Generator specifications
	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100,\
					"dropout": 0.3, "experiment": "Compas"}

	# Train the adversarial models for LIME with f and psi (fill te dictionary)
	adv_models["Perturbation"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain,\
								categorical_features=categorical_feature_indcs, feature_names=features, perturbation_multiplier=1)
	adv_models["DropoutVAE"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi(),
								generator = "DropoutVAE", generator_specs = generator_specs).train(xtrain, ytrain,\
								categorical_features=categorical_feature_indcs, integer_attributes = integer_attributes, feature_names=features, perturbation_multiplier=1)
	adv_models["RBF"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi(),\
								generator = "RBF", generator_specs = generator_specs).train(xtrain,\
								ytrain, feature_names=features, categorical_features=categorical_feature_indcs)
	adv_models["Forest"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi(),\
								generator = "Forest", generator_specs = generator_specs).train(xtrain,\
								ytrain, feature_names=features, categorical_features=categorical_feature_indcs)

	# Fill the dictionary with explanation methods
	for generator in ["Perturbation", "DropoutVAE", "RBF", "Forest"]:
		adv_explainers[generator] = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_models[generator].get_column_names(),\
								discretize_continuous=False,categorical_features=categorical_feature_indcs, generator=generator,\
								generator_specs=generator_specs, dummies=dummy_indcs, integer_attributes=integer_attributes)

	# We check every combination of adversarial model/explanation method
	for explainer in adv_explainers:
		adv_explainer = adv_explainers[explainer]
		for model in adv_models:
			adv_lime = adv_models[model]
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
			file_name = f"../Results/CompasLime/compasLimeSummary_adversarial_{model}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])
	
	# Repeat the same thing for two features (innocuous_model_psi_two is used)
	adv_models = dict()
	adv_explainers = dict()

	# Generator specifications
	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.3, "experiment": "Compas"}

	# Train the adversarial models for LIME with f and psi (fill te dictionary)
	adv_models["Perturbation"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain,\
								categorical_features=categorical_feature_indcs, feature_names=features, perturbation_multiplier=1)
	adv_models["DropoutVAE"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two(),
								generator = "DropoutVAE", generator_specs = generator_specs).train(xtrain, ytrain,\
								categorical_features=categorical_feature_indcs, integer_attributes = integer_attributes, feature_names=features, perturbation_multiplier=1)
	adv_models["RBF"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two(),\
								generator = "RBF", generator_specs = generator_specs).train(xtrain,\
								ytrain, feature_names=features, categorical_features=categorical_feature_indcs)
	adv_models["Forest"] = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi_two(),\
								generator = "Forest", generator_specs = generator_specs).train(xtrain,\
								ytrain, feature_names=features, categorical_features=categorical_feature_indcs)

	# Fill the dictionary with explanation methods
	for generator in ["Perturbation", "DropoutVAE", "RBF", "Forest"]:
		adv_explainers[generator] = lime.lime_tabular.LimeTabularExplainer(xtrain, feature_names=adv_models[generator].get_column_names(),
										discretize_continuous=False, categorical_features=categorical_feature_indcs, generator=generator,\
										generator_specs=generator_specs, dummies=dummy_indcs, integer_attributes=integer_attributes)

	# We check every combination of adversarial model/explanation method
	for explainer in adv_explainers:
		adv_explainer = adv_explainers[explainer]
		for model in adv_models:
			adv_lime = adv_models[model]
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
			file_name = f"../Results/CompasLime/compasLimeSummary2_adversarial_{model}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])

if __name__ == "__main__":
	experiment_main()
