# Robust-LIME-SHAP-and-IME

This is the implementation of methods gLIME, gSHAP and gIME for tabular data (see [[1]](#1) for description of the methods). Methods use three different data generators. Variational autoencoder using Monte Carlo dropout is implemented in Python and is located in folder Generators. The code was downloaded from [[5]](#5). rbfDataGen and treeEnsemble are not yet implemented in Python, so their R library semiArtificial[[6]](#6) implementation is used.

## Experiment

The code for experiment, conducted in [[1]](#1) can be found in folder Fooling-LIME-SHAP. The folder is a modified version of the original code from Slack [[4]](#4). To run experiment by yourself, run the selected file (they are named with the explanation method tested in the experiment and the dataset on which the experiment was conducted, in files containing "ime_variance" in the name is code for IME convergence rate experiment.) If you are using generators rbfDataGen and treeEnsemble, you have to generate the data in R (you can use R scripts from folder "R code". At the beginning of every experiment the execution stops after saving the training set, so you have the time to generate data in R.

For IME convergence rate experiment, trained classifiers and true Shapley values are uploaded to the repository, so you don't have to train/calculate them by yourself (this can take some time) and can be found in folder "Data/IME".

Besides the experiments, we also updated the Adversarial Models so they now support the use of data generators. To use the generator inside the Adversarial Model, you have to add argument "generator" when creating a new Adversarial Model. Possible values for this argument are: "DropoutVAE", "Forest" and "RBF". If this argument is not given, basic version of Adversarial Model from [[4]](#4) is used. If you use generator inside an adversarial models, you also have to give argument "generator_specs" to the constructor of Adversarial Model. This argument should be dictionary containing generator properties (for MCD-VAE this means its architecture and dropout rate, for treeEnsemble and rbfDataGen the name of the dataset on which the experiment is conducted).

### Example of training of adversarial model with generator

Let us assume we have following racist and unbiased model:

```python
# This is model b in paper
class racist_model_f:
    # Decision rule: classify negatively if race is black
    def predict(self,X):
        return np.array([0 if x[race_indc] > 0 else 1 for x in X])

    def predict_proba(self, X): 
        return one_hot_encode(self.predict(X))

    def score(self, X,y):
        return np.sum(self.predict(X)==y) / len(X)

# This is model psi in paper
class innocuous_model_psi:
    # Decision rule: classify according to randomly drawn column 'unrelated column'
    def predict_proba(self, X): 
        return one_hot_encode(np.array([0 if x[unrelated_indcs] > 0 else 1 for x in X]))
```

Then we can create adversarial model for method LIME on COMPAS dataset with following code:

```python
# Import adversarial models
from adversarial_models import *

# Generator specs for MCD-VAE (assuming that training set is stored in xtrain)
dvae_specs = {
    "original_dim": xtrain.shape[1], # Input layer of MCD-VAE
    "intermediate_dim": 16, # Hidden layer of encoder and decoder
    "latent_dim": xtrain.shape[1] // 2, # Dimension of latent space (output of encoder and input of decoder)
    "dropout": 0.3, # The probability of dropping a neuron in forward pass through the decoder
    "epochs": 1000 # The number of epochs during the training of the adversarial model
}

# Create adversarial model that uses MCD-VAE
adv_lime_dvae = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi(), generator = "DropoutVAE", generator_specs = dvae_specs)

# Generator specs for treeEnsemble
forest_specs = {
    "experiment": "Compas" # Dataset on which experiment is conducted (needed because data is generated in R)
}

# Create adversarial model that uses treeEnsemble
adv_lime_forest = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi(), generator = "Forest", generator_specs = forest_specs)

```

For training of the adversarial model that uses MCD-VAE, it is recommended to give the training function the indices of integer features in dataset (so after the data is generated with MCD-VAE, those features could be rounded) and perturbation_multiplier (integer that tells how many new samples are generated for each instance in training proces). For training of all adversarial models it is recommended to give the training function the least of features' names and the list of categorical features' indices. We will assume that we have all of this already stored (see experiment code if you are interested in how to get these values from dataset).

```python
# Training of the adversarial model that uses MCD-VAE (We assume that training set is stored in xtrain and that target variable is stored in ytrain).
adv_lime_dvae.train(xtrain, ytrain, categorical_features=categorical_feature_indcs, integer_attributes = integer_attributes, feature_names=features, perturbation_multiplier=1)

# Training of the adversarial model that uses treeEnsemble (We assume that training set is stored in xtrain and that target variable is stored in ytrain).
adv_lime_forest.train(xtrain, ytrain, categorical_features=categorical_feature_indcs, feature_names=features)
```

## gLIME

Method gLIME is implemented in file lime/lime_tabular.py. It is a modified version of method LIME from Python lime library [[2]](#2).

## gSHAP

Method gSHAP is implemented in file shap/explainers/kernel.py. It is a modified version of method Kernel SHAP [[8]](#8) from Python shap library [[3]](#3).

## gIME

Method gIME is implemented in file shap/explainers/sampling.py. It is a modified version of method IME [[7]](#7) included in Python shap library [[3]](#3).

## References
<a id="1">[1]</a>
Anonymous (2020)
Better sampling in explanation methods can prevent dieselgate-like deception
Submitted to International Conference on Learning Representations
https://openreview.net/forum?id=s0Chrsstpv2

<a id="2">[2]</a>
Ribeiro, M.
LIME
https://github.com/marcotcr/lime

<a id="3">[3]</a>
Lundberg, S.
SHAP
https://github.com/slundberg/shap

<a id="4">[4]</a>
Slack, D.
Fooling LIME and SHAP
https://github.com/dylan-slack/Fooling-LIME-SHAP

<a id="5">[5]</a>
Miok, K.
MCD-VAE
https://github.com/KristianMiok/MCD-VAE

<a id="6">[6]</a>
Robnik-Šikonja, M.
semiArtificial
https://CRAN.R-project.org/package=semiArtificial

<a id="7">[7]</a>
Štrumbelj, Erik and Kononenko, Igor (2013)
Explaining prediction models and individual predictions with feature contributions
Knowledge and Information Systems, 41, 647-665

<a id="7">[7]</a>
Štrumbelj, E. and Kononenko, I. (2013)
Explaining prediction models and individual predictions with feature contributions
Knowledge and Information Systems, 41, 647-665

<a id="8">[8]</a>
Lundberg, S. M. and Lee, S. (2017)
A Unified Approach to Interpreting Model Predictions
Advances in Neural Information Processing Systems 30, 4765-4774

<a id="9">[9]</a>
Ribeiro, M. T. and Singh, S. and Guestrin, C. (2016)
"Why Should I Trust You?": Explaining the Predictions of Any Classifier
Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 1135-1144