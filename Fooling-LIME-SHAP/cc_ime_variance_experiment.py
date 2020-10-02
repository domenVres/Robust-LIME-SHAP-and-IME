"""
Experiment that compares the IME convergence rate on CC when basic perturbation sampling, MCD-VAE and
treeEnsemble with data fill are used
"""

import pandas as pd
import numpy as np

from utils import *
from get_data import *

import shap

import pickle

xtrain = pd.read_csv("..\Data\IME\cc_train.csv")
xtest = pd.read_csv("..\Data\IME\cc_test.csv")
ytrain = xtrain.pop("y")
ytest = xtest.pop("y")

features = [c for c in xtrain]

# Only categorical features are unrelated columns
categorical_features = ["unrelated_column_one", "unrelated_column_two"]
categorical_idcs = [features.index("unrelated_column_one"), features.index("unrelated_column_two")]
dummy_indcs = [[categorical_idcs[0]], [categorical_idcs[1]]]

# Find integer features
integer_attributes = [i for i, feature in enumerate(xtrain.columns)
					if (xtrain[feature].dtype in ["int64", "int32", "int8", "uint64", "uint32", "uint8"] and i not in categorical_idcs)]

xtrain = xtrain.values
xtest = xtest.values

latent_dim = xtrain.shape[1] // 2
original_dim = xtrain.shape[1]

"""
Naive Bayes
"""
print ('---------------------')
print ('Beginning with Naive Bayes')
print ('---------------------')

# Load classifier and true Shapley values
bayes = pickle.load(open("../Data/IME/CC/bayes_model.sav", 'rb'))
shapley_values = np.load("../Data/IME/CC/bayes_values.npy")

generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.3,\
                    "experiment": "CC", "feature_names": features}

perturbation_explainer = shap.SamplingExplainer(bayes.predict, xtrain)
dvae_explainer = shap.SamplingExplainer(bayes.predict, xtrain, generator="DropoutVAE", generator_specs=generator_specs,\
                            dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=100)
forest_explainer = shap.SamplingExplainer(bayes.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
                            dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=100)

# Setup experiment
perturbation_explainer.create_experiment_table(shapley_values)
dvae_explainer.create_experiment_table(shapley_values)
forest_explainer.create_experiment_table(shapley_values)

# Experiment for perturbations
perturbation_explainer.shap_values(xtest, nsamples="variance", is_experiment = True)
perturbation_data = perturbation_explainer.get_experiment_dataframe()

# Save dataframe for perturbation
perturbation_data.to_csv("../Results/CCImeVariance/perturbation_bayes.csv", index = False)

# Experiment for MCD-VAE
dvae_explainer.shap_values(xtest, nsamples="variance", is_experiment = True)
dvae_data = dvae_explainer.get_experiment_dataframe()

# Save dataframe for MCD-VAE
dvae_data.to_csv("../Results/CCImeVariance/dvae_bayes.csv", index = False)

# Experiment for ForestFill
forest_explainer.shap_values(xtest, nsamples="variance", is_experiment = True, fill_data=True, data_location="..\Data/IME/cc_forest_generated.csv", distribution_size=1000)
dvae_data = forest_explainer.get_experiment_dataframe()

# Save dataframe for ForestFill
dvae_data.to_csv("../Results/CCImeVariance/forest_bayes.csv", index = False)

"""
Linear SVM
"""
print ('---------------------')
print ('Beginning with Linear SVM')
print ('---------------------')

# Load classifier and true Shapley values
svm = pickle.load(open("../Data/IME/CC/svm_model.sav", 'rb'))
shapley_values = np.load("../Data/IME/CC/svm_values.npy")

generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.3,\
                    "experiment": "CC", "feature_names": features}

perturbation_explainer = shap.SamplingExplainer(svm.predict, xtrain)
dvae_explainer = shap.SamplingExplainer(svm.predict, xtrain, generator="DropoutVAE", generator_specs=generator_specs,\
                            dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=100)
forest_explainer = shap.SamplingExplainer(svm.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
                            dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=100)

# Setup experiment
perturbation_explainer.create_experiment_table(shapley_values)
dvae_explainer.create_experiment_table(shapley_values)
forest_explainer.create_experiment_table(shapley_values)

# Experiment for perturbations
perturbation_explainer.shap_values(xtest, nsamples="variance", is_experiment = True)
perturbation_data = perturbation_explainer.get_experiment_dataframe()

# Save dataframe for perturbation
perturbation_data.to_csv("../Results/CCImeVariance/perturbation_svm.csv", index = False)

# Experiment for MCD-VAE
dvae_explainer.shap_values(xtest, nsamples="variance", is_experiment = True)
dvae_data = dvae_explainer.get_experiment_dataframe()

# Save dataframe for MCD-VAE
dvae_data.to_csv("../Results/CCImeVariance/dvae_svm.csv", index = False)

# Experiment for ForestFill
forest_explainer.shap_values(xtest, nsamples="variance", is_experiment = True, fill_data=True, data_location="..\Data/IME/cc_forest_generated.csv", distribution_size=1000)
dvae_data = forest_explainer.get_experiment_dataframe()

# Save dataframe for ForestFill
dvae_data.to_csv("../Results/CCImeVariance/forest_svm.csv", index = False)

"""
Random Forest
"""
print ('---------------------')
print ('Beginning with Random Forest')
print ('---------------------')

# Load classifier and true Shapley values
forest = pickle.load(open("../Data/IME/CC/forest_model.sav", 'rb'))
shapley_values = np.load("../Data/IME/CC/forest_values.npy")

generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.3,\
                    "experiment": "CC", "feature_names": features}

perturbation_explainer = shap.SamplingExplainer(forest.predict, xtrain)
dvae_explainer = shap.SamplingExplainer(forest.predict, xtrain, generator="DropoutVAE", generator_specs=generator_specs,\
                            dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=100)
forest_explainer = shap.SamplingExplainer(forest.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
                            dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=100)

# Setup experiment
perturbation_explainer.create_experiment_table(shapley_values)
dvae_explainer.create_experiment_table(shapley_values)
forest_explainer.create_experiment_table(shapley_values)

# Experiment for perturbations
perturbation_explainer.shap_values(xtest, nsamples="variance", is_experiment = True)
perturbation_data = perturbation_explainer.get_experiment_dataframe()

# Save dataframe for perturbation
perturbation_data.to_csv("../Results/CCImeVariance/perturbation_forest.csv", index = False)

# Experiment for MCD-VAE
dvae_explainer.shap_values(xtest, nsamples="variance", is_experiment = True)
dvae_data = dvae_explainer.get_experiment_dataframe()

# Save dataframe for MCD-VAE
dvae_data.to_csv("../Results/CCImeVariance/dvae_forest.csv", index = False)

# Experiment for ForestFill
forest_explainer.shap_values(xtest, nsamples="variance", is_experiment = True, fill_data=True, data_location="..\Data/IME/cc_forest_generated.csv", distribution_size=1000)
dvae_data = forest_explainer.get_experiment_dataframe()

# Save dataframe for ForestFill
dvae_data.to_csv("../Results/CCImeVariance/forest_forest.csv", index = False)

"""
Neural Network
"""
print ('---------------------')
print ('Beginning with Neural Network')
print ('---------------------')

# Load classifier and true Shapley values
network = pickle.load(open("../Data/IME/CC/nn_model.sav", 'rb'))
shapley_values = np.load("../Data/IME/CC/nn_values.npy")

generator_specs = {"original_dim": original_dim, "intermediate_dim": 8, "latent_dim": latent_dim, "epochs": 100, "dropout": 0.3,\
                    "experiment": "CC", "feature_names": features}

perturbation_explainer = shap.SamplingExplainer(network.predict, xtrain)
dvae_explainer = shap.SamplingExplainer(network.predict, xtrain, generator="DropoutVAE", generator_specs=generator_specs,\
                            dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=100)
forest_explainer = shap.SamplingExplainer(network.predict, xtrain, generator="Forest", generator_specs=generator_specs,\
                            dummy_idcs=dummy_indcs, integer_idcs=integer_attributes, instance_multiplier=100)

# Setup experiment
perturbation_explainer.create_experiment_table(shapley_values)
dvae_explainer.create_experiment_table(shapley_values)
forest_explainer.create_experiment_table(shapley_values)

# Experiment for perturbations
perturbation_explainer.shap_values(xtest, nsamples="variance", is_experiment = True)
perturbation_data = perturbation_explainer.get_experiment_dataframe()

# Save dataframe for perturbation
perturbation_data.to_csv("../Results/CCImeVariance/perturbation_nn.csv", index = False)

# Experiment for MCD-VAE
dvae_explainer.shap_values(xtest, nsamples="variance", is_experiment = True)
dvae_data = dvae_explainer.get_experiment_dataframe()

# Save dataframe for MCD-VAE
dvae_data.to_csv("../Results/CCImeVariance/dvae_nn.csv", index = False)

# Experiment for ForestFill
forest_explainer.shap_values(xtest, nsamples="variance", is_experiment = True, fill_data=True, data_location="..\Data/IME/cc_forest_generated.csv", distribution_size=1000)
dvae_data = forest_explainer.get_experiment_dataframe()

# Save dataframe for ForestFill
dvae_data.to_csv("../Results/CCImeVariance/forest_nn.csv", index = False)