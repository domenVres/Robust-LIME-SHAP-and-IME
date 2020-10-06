"""
The IME experiment MAIN for COMPAS.
 * Run the file and the COMPAS experiments will complete
 * This may take some time because we iterate through every instance in the test set for
   both LIME and SHAP explanations take some time to compute
 * The print outs can be interpreted as maps from the RANK to the rate at which the feature occurs in the rank.. e.g:
 	    1: [('length_of_stay', 0.002592352559948153), ('unrelated_column_one', 0.9974076474400518)]
   can be read as the first unrelated column occurs ~100% of the time in as the most important feature
 * "Nothing shown" refers to IME yielding only 0 shapley values 
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
params = Params("../Fooling-LIME-SHAP-master/model_configurations/experiment_params.json")
np.random.seed(params.seed)

# Prepare data (use this option if you are not using treeEnsemble)
'''
X, y, cols = get_and_preprocess_compas_data_RBF(params)

# add unrelated columns, setup
X['unrelated_column_one'] = np.random.choice([0,1],size=X.shape[0])
X['unrelated_column_two'] = np.random.choice([0,1],size=X.shape[0])

data_train, data_test, ytrain, ytest = train_test_split(X, y, test_size=0.1)
'''

# If data was split before the experiment (required for treeEnsemble)
data_train = pd.read_csv("..\Data\compas_forest_train.csv")
data_test = pd.read_csv("..\Data\compas_forest_test.csv")
ytrain = data_train.pop("response")
ytest = data_test.pop("response")

# Convert categorical features to one-hot encoded vector (this must be done after saving the data
# as rbfDataGen and treeEnsemble work with original values)
data_test = pd.get_dummies(data_test)

features = [c for c in data_test]

categorical_feature_name = ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M',\
                            'sex_Female', 'sex_Male', 'race', 'unrelated_column_one', 'unrelated_column_two']
categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]
dummy_indcs = [[categorical_feature_indcs[0]], [categorical_feature_indcs[1], categorical_feature_indcs[2]],\
            [categorical_feature_indcs[3], categorical_feature_indcs[4]], [categorical_feature_indcs[5]],\
			[categorical_feature_indcs[6]], [categorical_feature_indcs[7]]]

# Find integer features
integer_attributes = [i for i, feature in enumerate(data_test.columns)
					if (data_test[feature].dtype in ["int64", "int32", "int8", "uint64", "uint32", "uint8"] and i not in categorical_feature_indcs)]

latent_dim = data_train.shape[1] // 2
data_train = pd.get_dummies(data_train)

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
	Run through experiments for IME on compas using both one and two unrelated features.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""
	print ('---------------------')
	print ('Beginning IME COMPAS Experiments....')
	print ('---------------------')

	################################################
	# One unrelated (ininnocuous_model_psi is used)
	################################################
	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.2,\
						"experiment": "Compas", "feature_names": features}

	print ('---------------------')
	print ('Training adversarial models....')
	print ('---------------------')
	# Adversarial models
	adv_models = dict()
	adv_models["Perturbation"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain,\
														feature_names=features, perturbation_multiplier = 1)
	adv_models["DropoutVAE"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi(), generator = "DropoutVAE", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, perturbation_multiplier = 1)
	adv_models["ForestFill"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi(), generator = "Forest", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, perturbation_multiplier = 1)

	for adversarial in ["Perturbation", "DropoutVAE", "ForestFill"]:
		print ('---------------------')
		print (f'Training explainers with adversarial {adversarial}....')
		print ('---------------------')

		adv_model = adv_models[adversarial]

		# Explainers
		adv_explainers = dict()
		adv_explainers["Perturbation"] = shap.SamplingExplainer(adv_model.predict, xtrain)
		adv_explainers["DropoutVAE"] = shap.SamplingExplainer(adv_model.predict, xtrain, generator="DropoutVAE", generator_specs=generator_specs,\
								dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=1000)
		adv_explainers["ForestFill"] = shap.SamplingExplainer(adv_model.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
								dummy_idcs=dummy_indcs, integer_idcs=integer_attributes)

		for explainer in ["Perturbation", "DropoutVAE", "ForestFill"]:
			adv_explainer = adv_explainers[explainer]
			explanations = adv_explainer.shap_values(xtest, fill_data=True, data_location="...\Data/compas_forest_ime.csv", distribution_size=1000)

			# format for display
			formatted_explanations = []
			for exp in explanations:
				formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

			print (f"IME Ranks and Pct Occurances one unrelated feature, adversarial: {adversarial}, explainer: {explainer}:")
			summary = experiment_summary(formatted_explanations, features)
			print (summary)
			print ("Fidelity:",round(adv_model.fidelity(xtest),2))

			file_name = f"../Results/CompasIme/compasImeSummary_adversarial_{adversarial}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])

	####################################################
	# Two unrelated (ininnocuous_model_psi_two is used)
	####################################################
	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.2,\
						"experiment": "Compas", "feature_names": features}

	print ('---------------------')
	print ('Training adversarial models....')
	print ('---------------------')
	# Adversarial models
	adv_models = dict()
	adv_models["Perturbation"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi_two()).train(xtrain, ytrain,\
														feature_names=features, perturbation_multiplier=1)
	adv_models["DropoutVAE"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi_two(), generator = "DropoutVAE", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, perturbation_multiplier=1)
	adv_models["ForestFill"] = Adversarial_IME_Model(racist_model_f(), innocuous_model_psi_two(), generator = "Forest", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, perturbation_multiplier=1)

	for adversarial in ["Perturbation", "DropoutVAE", "ForestFill"]:
		adv_model = adv_models[adversarial]

		print ('---------------------')
		print (f'Training explainers with adversarial {adversarial}....')
		print ('---------------------')
		# Explainers
		adv_explainers = dict()
		adv_explainers["Perturbation"] = shap.SamplingExplainer(adv_model.predict, xtrain)
		adv_explainers["DropoutVAE"] = shap.SamplingExplainer(adv_model.predict, xtrain, generator="DropoutVAE", generator_specs=generator_specs,\
								dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=1000)
		adv_explainers["ForestFill"] = shap.SamplingExplainer(adv_model.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
								dummy_idcs=dummy_indcs, integer_idcs=integer_attributes)

		for explainer in ["Perturbation", "DropoutVAE", "ForestFill"]:
			adv_explainer = adv_explainers[explainer]
			explanations = adv_explainer.shap_values(xtest, fill_data=True, data_location="...\Data/compas_forest_ime.csv", distribution_size=1000)

			# format for display
			formatted_explanations = []
			for exp in explanations:
				formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])

			print (f"IME Ranks and Pct Occurances two unrelated features, adversarial: {adversarial}, explainer: {explainer}:")
			summary = experiment_summary(formatted_explanations, features)
			print (summary)
			print ("Fidelity:",round(adv_model.fidelity(xtest),2))

			file_name = f"../Results/CompasIme/compasImeSummary2_adversarial_{adversarial}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])
	print ('---------------------')

if __name__ == "__main__":
	experiment_main()
