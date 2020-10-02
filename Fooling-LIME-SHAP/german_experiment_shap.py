"""
The SHAP experiment MAIN for GERMAN.
"""
import warnings
warnings.filterwarnings('ignore') 

from adversarial_models import * 
from utils import *
from get_data import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pandas as pd

import lime
import lime.lime_tabular
import shap

from sklearn.cluster import KMeans 

from copy import deepcopy

import csv

''' Function that helps to determine the number of clusters with silhouette score
X -> numpy array, data
n_clusters_list -> list of candidates for number of clusters
'''
def s_score(X, n_clusters_list):
	for n_clusters in n_clusters_list:
		# Create a subplot with 1 row and 2 columns
		fig, ax = plt.subplots(1, 1)
		fig.set_size_inches(18, 7)

		# The 1st subplot is the silhouette plot
		# The silhouette coefficient can range from -1, 1 but in this example all
		# lie within [-0.1, 1]
		ax.set_xlim([-0.1, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

		# Clustering
		k_means = KMeans(n_clusters=n_clusters)
		clusters = k_means.fit_predict(X)

		# Average silhouette score
		silhouette_avg = silhouette_score(X, clusters)
		# Silhouette score for every instance
		silhouette_values = silhouette_samples(X, clusters)

		# Start with y=10, so there are spaces between clusters on graph
		y_lower = 10
		for i in range(n_clusters):
			cluster_silhouette_values = silhouette_values[clusters == i]
			cluster_silhouette_values.sort()
			
			# Upper bound for y
			y_upper = y_lower + cluster_silhouette_values.shape[0]
			# Color of the cluster
			color = cm.nipy_spectral(float(i) / n_clusters)
			# Fill the figure
			ax.fill_betweenx(np.arange(y_lower, y_upper),
							0, cluster_silhouette_values,
							facecolor=color, edgecolor=color, alpha=0.7)
			
			# Label the silhouette plots with their cluster numbers at the middle
			ax.text(-0.05, y_lower + 0.5 * cluster_silhouette_values.shape[0], str(i))

			# Increase lower bound for y so next cluster is 10 above this one
			y_lower = y_upper + 10

		ax.set_title(f"Display of silhouette scores for {n_clusters} clusters")
		ax.set_xlabel("Silhouette score")
		ax.set_ylabel("Cluster label")

		# The vertical line for average silhouette score of all the values
		ax.axvline(x=silhouette_avg, color="red", linestyle="--")

		ax.set_yticks([])  # Clear the yaxis labels / ticks
		ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

	plt.show()

# Set up experiment parameters
params = Params("model_configurations/experiment_params.json")

# Prepare data (use this option if you are not using treeEnsemble with data fill)
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

# If data was split before the experiment (required for treeEnsemble with data fill)
data_train = pd.read_csv("..\Data\german_forest_train.csv")
data_test = pd.read_csv("..\Data\german_forest_test.csv")
ytrain = data_train.pop("response")
ytest = data_test.pop("response")

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
data_train = pd.get_dummies(data_train)
features = [c for c in data_train]

# Names without dummy variables (they are used for explanation methods with data generators, because dummy variables are grouped there)
original_names = [c for c in data_train]
original_names[original_names.index("CheckingAccountBalance_geq_200")] = "CheckingAccountBalance"
original_names.remove('CheckingAccountBalance_geq_0_lt_200')
original_names.remove('CheckingAccountBalance_lt_0')
original_names[original_names.index('SavingsAccountBalance_geq_500')] = "SavingsAccountBalance"
original_names.remove('SavingsAccountBalance_geq_100_lt_500')
original_names.remove('SavingsAccountBalance_lt_100')
original_names[original_names.index('YearsAtCurrentJob_geq_4')] = "YearsAtCurrentJob"
original_names.remove('YearsAtCurrentJob_geq_1_lt_4')
original_names.remove('YearsAtCurrentJob_lt_1')

# So user can check original_names
print(original_names)
input("Please check original feature names. Press enter to continue.")

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
	Run through experiments for SHAP on german.
	* This may take some time given that we iterate through every point in the test set
	* We print out the rate at which features occur in the top three features
	"""

	# Setup SHAP

	# Choose the optimal number of clusters
	candidates = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
	s_score(xtrain, candidates)

	n_clusters = int(input("Please enter the optimal number of clusters: "))

	background_distribution = shap.kmeans(xtrain, n_clusters)
	generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.2,\
						"experiment": "German", "feature_names": features}

	# Adversarial models
	adv_models = dict()
	adv_models["Perturbation"] = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi()).train(xtrain, ytrain, feature_names=features)
	adv_models["DropoutVAE"] = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi(), generator = "DropoutVAE", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_idcs, integer_idcs=integer_attributes, n_samples=10*xtrain.shape[0])
	adv_models["RBF"] = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi(), generator = "RBF", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_idcs, integer_idcs=integer_attributes)
	adv_models["Forest"] = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi(), generator = "Forest", generator_specs = generator_specs).\
            				train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_idcs, integer_idcs=integer_attributes)

	for adversarial in ["Perturbation", "DropoutVAE", "RBF", "Forest"]:
		adv_shap = adv_models[adversarial]

		# Explainers
		adv_kernel_explainers = dict()
		adv_kernel_explainers["Perturbation"] = shap.KernelExplainer(adv_shap.predict, background_distribution)
		adv_kernel_explainers["DropoutVAE"] = shap.KernelExplainer(adv_shap.predict, xtrain, generator="DropoutVAE", generator_specs=generator_specs,\
								dummy_idcs=dummy_idcs, integer_idcs=integer_attributes)
		adv_kernel_explainers["RBF"] = shap.KernelExplainer(adv_shap.predict, xtrain, generator="RBF", generator_specs=generator_specs,\
								dummy_idcs=dummy_idcs)
		adv_kernel_explainers["Forest"] = shap.KernelExplainer(adv_shap.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
								dummy_idcs=dummy_idcs)
		adv_kernel_explainers["ForestFill"] = shap.KernelExplainer(adv_shap.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
								dummy_idcs=dummy_idcs)

		for explainer in ["Perturbation", "DropoutVAE", "RBF", "Forest", "ForestFill"]:
			adv_kernel_explainer = adv_kernel_explainers[explainer]
			
			# Fill data option
			if explainer == "ForestFill":
				explanations = adv_kernel_explainer.shap_values(xtest, fill_data=True, data_location="..\Data/german_forest_shap.csv")

			else:
				explanations = adv_kernel_explainer.shap_values(xtest)

			# format for display
			formatted_explanations = []
			for exp in explanations:
				if explainer == "Perturbation":
					formatted_explanations.append([(features[i], exp[i]) for i in range(len(exp))])
				else:
					formatted_explanations.append([(original_names[i], exp[i]) for i in range(len(exp))])

			print (f"SHAP Ranks and Pct Occurances one unrelated feature, adversarial: {adversarial}, explainer: {explainer}:")
			if explainer == "Perturbation":
				summary = experiment_summary(formatted_explanations, features)
			else:
				summary = experiment_summary(formatted_explanations, original_names)
			print (summary)
			print ("Fidelity:",round(adv_shap.fidelity(xtest),2))

			file_name = f"../Results/GermanShap/germanShapSummary_adversarial_{adversarial}_explainer_{explainer}.csv"
			with open(file_name, "w") as output:
				w = csv.writer(output)
				for key, val in summary.items():
					w.writerow([key] + [pair for pair in val])

if __name__ == "__main__":
	experiment_main()
