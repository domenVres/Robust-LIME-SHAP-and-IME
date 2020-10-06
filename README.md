# Robust-LIME-SHAP-and-IME

This is the implementation of methods gLIME, gSHAP and gIME for tabular data (see [[1]](#1) for description of the methods). Methods use three different data generators. Variational autoencoder using Monte Carlo dropout is implemented in Python and is located in folder Generators. The code was downloaded from [[5]](#5). rbfDataGen and treeEnsemble are not yet implemented in Python, so their R library semiArtificial [[6]](#6) implementation is used.

## Experiment

The code for experiment, conducted in [[1]](#1) can be found in folder Fooling-LIME-SHAP. The folder is a modified version of the original code from Slack [[4]](#4). To run experiment by yourself, run the selected file (they are named with the explanation method tested in the experiment and the dataset on which the experiment was conducted). In files that contain "ime_variance" in their name is code for IME convergence rate experiment. If you are using generators rbfDataGen and treeEnsemble, you have to generate the data in R (you can use R scripts from folder "R code"). At the beginning of every experiment the execution stops after saving the training set, so you have the time to generate data in R.

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

Then we can create adversarial models for method LIME on COMPAS dataset with following code:

```python
# Import adversarial models
from adversarial_models import *

# Generator specs for MCD-VAE (assuming that training set is stored in xtrain)
dvae_specs = {
    "original_dim": xtrain.shape[1], # Input layer of MCD-VAE
    "intermediate_dim": 16, # Hidden layer of encoder and decoder
    "latent_dim": xtrain.shape[1] // 2, # Dimension of latent space (output of encoder and input of decoder)
    "dropout": 0.3, # The probability of dropping a neuron in forward pass through the decoder
    "epochs": 100 # The number of epochs during the training of the adversarial model
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

For training of the adversarial model that uses MCD-VAE, it is recommended to give the training function the indices of integer features in dataset (so after the data is generated with MCD-VAE, those features could be rounded) and perturbation_multiplier (integer that tells how many new samples are generated for each instance in training proces). For training of all adversarial models it is recommended to give the training function the list of features' names and the list of categorical features' indices. In following code block we assume that we have all of this already stored (see experiment code if you are interested in how to get these values from dataset).

```python
# Training of the adversarial model that uses MCD-VAE (We assume that training set is stored in xtrain and that target variable is stored in ytrain).
adv_lime_dvae.train(xtrain, ytrain, categorical_features=categorical_feature_indcs, integer_attributes = integer_attributes, feature_names=features, perturbation_multiplier=1)

# Training of the adversarial model that uses treeEnsemble (We assume that training set is stored in xtrain and that target variable is stored in ytrain).
adv_lime_forest.train(xtrain, ytrain, categorical_features=categorical_feature_indcs, feature_names=features)
```

## gLIME

Original implementation of method LIME that uses basic perturbation sampling is taken from [[2]](#2). We altered the version of the method for the tabular data (file "lime/lime_tabular.py") so generators MCD-VAE, rbfDataGen and treeEnsemble can be used for the sampling in the method (see [[1]](#1) for more details).

To use the data generator inside the LIME method, the argument generator (possible values: "DropoutVAE", "RBF" and "Forest") has to be given to the constructor of LIME model (generator is set to "Perturb" by default, which represents the basic version of the method LIME). If data generator is being used, you should also provide the dictionary with generator specifications as argument generator_specs (see the Experiment section for the details on dictionary values). When using MCD-VAE it is also recommended to provide the arguments dummies (a list of lists of indices that represent the same one-hot encoded feature) and integer_attributes (the list of indices of integer features) so the samples can be corrected properly after they are generated. When using rbfDataGen and treeEnsemble last two arguments are not necessary as the data is being generated in R.

### Example of use on COMPAS dataset

In following code blocks we assume that our data is preprocessed (race is converted to binary feature (black or not) and all categorical features are one-hot encoded, see experiment code for more details). We assume that training set is stored in xtrain, test set is stored in xtest and list of feature names is stored in features. We also assume we have adversarial models from Experiment section trained and generator specifications stored (see Experiment section).

Creation of the LIME model (training of the generators is also executed in the constructor) using data generators:

```python
# Import the lime models
import lime

# List of categorical features names and their indices
categorical_feature_name = ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M',\
                            'sex_Female', 'sex_Male', 'race', 'unrelated_column_one', 'unrelated_column_two']
categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]

# List of lists of indices representing the same one-hot encoded features as it should be provided to the LIME model
# (note that each of the lists stored in the main list contains the indices for one one-hot encoded feature)
dummy_indcs = [[categorical_feature_indcs[0]], [categorical_feature_indcs[1], categorical_feature_indcs[2]],\
            [categorical_feature_indcs[3], categorical_feature_indcs[4]], [categorical_feature_indcs[5]],\
            [categorical_feature_indcs[6]], [categorical_feature_indcs[7]]]

# A simple way to find the integer features (we add only those that are not categorical features, as they are
# handled in postprocessing the same way as one-hot encoded features)
integer_attributes = [i for i, feature in enumerate(data_test.columns)
                if (data_test[feature].dtype in ["int64", "int32", "int8", "uint64", "uint32", "uint8"] and i not in categorical_feature_indcs)]

# Generate LIME model that uses MCD-VAE for sampling
lime_dvae = lime.lime_tabular.LimeTabularExplainer(
                        xtrain, # Training set (numpy array)
                        feature_names=features, # List of features' names 
                        discretize_continuous=False, # See original lime repository or lime_tabular code for details
                        categorical_features=categorical_feature_indcs,
                        generator = "DropoutVAE", # MCD-VAE
                        generator_specs = dvae_specs, # See code from Experiment section
                        dummies=dummy_indcs,
                        integer_attributes=integer_attributes)

# Generate LIME model that uses treeEnsemble for sampling
lime_forest = lime.lime_tabular.LimeTabularExplainer(
                        xtrain, # Training set (numpy array)
                        feature_names=features, # List of features' names 
                        discretize_continuous=False, # See original lime repository or lime_tabular code for details
                        categorical_features=categorical_feature_indcs,
                        generator = "Forest", # treeEnsemble
                        generator_specs = forest_specs, # See code from Experiment section
                        )
```

The code for explaining instancest from test set is same for gLIME and LIME (see [[2]](#2) for more details). The whole test set can be explained with following code:

```python

# Explain the predictions of adversarial model that uses MCD-VAE on the test set with LIME that uses MCD-VAE
dvae_explanations = []
for i in range(xtest.shape[0]):
    dvae_explanations.append(lime_dvae.explain_instance(xtest[i], adv_lime_dvae.predict_proba).as_list())

# Explain the predictions of adversarial model that uses treeEnsemble on the test set with LIME that uses treeEnsemble
forest_explanations = []
for i in range(xtest.shape[0]):
    forest_explanations.append(lime_forest.explain_instance(xtest[i], adv_lime_forest.predict_proba).as_list())
```

## gSHAP

Method gSHAP represents the modified version of method Kernel SHAP from shap repository [[3]](#3). The code for this method is located in file shap/explainers/kernel.py. gSHAP supports the usage of generators MCD-VAE, rbfDataGen and treeEnsemble (two possibble versions: the one with fill_data set to False, generates the distribution set according to the whole training set and the one with fill_data set to True generates the distribution set around the incoming instance).

To use the data generator inside SHAP, the argument generator (possible values: "DropoutVAE", "RBF" and "Forest") has to be given to the constructor of Kernel SHAP model (generator is set to "Perturb" by default, which represents the basic version of Kernel SHAP). If data generator is being used, you should also provide the dictionary with generator specifications (argument generator_specs), list of lists of indices of one-hot encoded features (argument dummy_idcs) and list of indices of integer features (argument integer_idcs). See gLIME section for more details on everything listed above. Only difference to the values used in gLIME section is in generator specifications for treeEnsemble and rbfDataGen, where "feature_names" value also has to be provided. This is due to fact that the data is being one-hot encoded after is being loaded from R and we need to store the original names so we can put the features in the correct order. If you are using MCD-VAE, you can set the size of distribution size with the instance_multiplier argument (it is set to 100 by default).

### Example of use on COMPAS dataset

In following code blocks the same assumptions are made as in gLIME section. Additionally we assume that we have list of lists of indices of one-hot encoded features (variable dummy_indcs), list of categorical features' indices (variable ategorical_feature_indcs) and list of integer features (variable integer_attributes) already stored. Check code in gLIME section for details on those variables.

Creation of the Kernel SHAP model (training of the generators is also executed in the constructor) using data generators:

```python
# Import shap models
import shap

# We have to add the list of feature names (recall that it is stored in features) to the generator specifications of treeEnsemble
forest_specs["feature_names"] = features

# Create adversarial models for SHAP
adv_shap_dvae = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi(), generator = "DropoutVAE", generator_specs = dvae_specs).
            train(xtrain, ytrain, feature_names=features, dummy_idcs=dummy_indcs, integer_idcs=integer_attributes)
adv_shap_forest = Adversarial_Kernel_SHAP_Model(racist_model_f(), innocuous_model_psi(), generator = "Forest", generator_specs = forest_specs).
            train(xtrain, ytrain, feature_names=features)

# Create gSHAP model that MCD-VAE too explain predictions of adversarial shap model that uses MCD-VAE
shap_dvae = shap.KernelExplainer(
    adv_shap_dvae.predict, # Predict function of the model we are explaining has to be given to the Kernel SHAP constructor
    xtrain,
    generator="DropoutVAE",
    generator_specs=dvae_specs,
    dummy_idcs=dummy_indcs,
    integer_idcs=integer_attributes,
    instance_multiplier = 100 # Size of the distribution set
)

# Create gSHAP model that uses treeEnsemble too explain predictions of adversarial shap model that uses treeEnsemble
shap_forest = shap.KernelExplainer(
    adv_shap_forest.predict, # Predict function of the model we are explaining has to be given to the Kernel SHAP constructor
    xtrain, generator="Forest",
    generator_specs=generator_specs,
    dummy_idcs=dummy_indcs
)

```

The code for explaining instancest from test set is same for gSHAP and Kernel SHAP (see [[3]](#3) for more details), except when the treeEnsemble is being used with option to generate the distributon set around the incoming instance. In that case the argument fill_data (which is added in gSHAP) of function shap_values has to be set to True. In that case you also have to provide the path to the file containing in R generated distribution sets.

Aslo be careful when using gSHAP - if you provide the dummy_idcs to the constructor, the method will group the features that are in the same list in dummy_idcs (see [[3]](#3) for more details on grouping of features). Therefore the group that represents the same one-hot encoded feature will be given only one contribution (the contribution of the original feature). Check the experiment code for shap to see how features' names were adopted accordingly in formatted explanations.

Code for explaining the whole test set (with MCD-VAE and treeEnsemble using both options):

```python
# Explanations using MCD-VAE (the model we are explaining does not have to be provided this time as it was already
# given to the constructor of the Kernel SHAP model)
dvae_explanations = shap_dvae.shap_values(xtest)

# Explanations using treeEnsemble (fill_data is by default set to False so this option creates the distribution
# set based on the whole training set)
forest_explanations =  shap_forest.shap_values(xtest)

# Explanation using treeEnsemble with distribution set being generated around the incoming instance (fill_data
# has to be set to True and path to the generated distribution sets has to be provided)
forestFill_explanations = shap_forest.shap_values(xtest, fill_data=True, data_location="Data/compas_forest_shap.csv")
```

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