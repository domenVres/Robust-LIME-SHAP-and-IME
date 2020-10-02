import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from Generators.DropoutVAE import DropoutVAE

from copy import deepcopy

import shap

class Adversarial_Model(object):
	"""	A scikit-learn style adversarial explainer base class for adversarial models.  This accetps 
	a scikit learn style function f_obscure that serves as the _true classification rule_ for in distribution
	data.  Also, it accepts, psi_display: the classification rule you wish to display by explainers (e.g. LIME/SHAP).
	Ideally, f_obscure will classify individual instances but psi_display will be shown by the explainer.

	Parameters
	----------
	f_obscure : function
	psi_display : function
	"""
	def __init__(self, f_obscure, psi_display):
		self.f_obscure = f_obscure
		self.psi_display = psi_display

		self.cols = None
		self.scaler = None
		self.numerical_cols = None

	def predict_proba(self, X, threshold=0.5):
		""" Scikit-learn style probability prediction for the adversarial model.  

		Parameters
		----------
		X : np.ndarray

		Returns
		----------
		A numpy array of the class probability predictions of the advesarial model.
		"""
		if self.perturbation_identifier is None:
			raise NameError("Model is not trained yet, can't perform predictions.")

		# generate the "true" predictions on the data using the "bad" model -- this is b in the paper
		predictions_to_obscure = self.f_obscure.predict_proba(X)

		# generate the "explain" predictions -- this is psi in the paper

		predictions_to_explain_by = self.psi_display.predict_proba(X)

		# in the case that we're only considering numerical columns
		if self.numerical_cols:
			X = X[:,self.numerical_cols]

		# allow thresholding for finetuned control over psi_display and f_obscure
		pred_probs = self.perturbation_identifier.predict_proba(X)
		perturbation_preds = (pred_probs[:,1] >= threshold)
		sol = np.where(np.array([perturbation_preds == 1,perturbation_preds==1]).transpose(), predictions_to_obscure, predictions_to_explain_by)

		return sol

	def predict(self, X):
		"""	Scikit-learn style prediction. Follows from predict_proba.

		Parameters
		----------
		X : np.ndarray
		
		Returns
		----------
		A numpy array containing the binary class predictions.
		"""
		pred_probs = self.predict_proba(X)
		return np.argmax(pred_probs,axis=1)

	def score(self, X_test, y_test):	
		""" Scikit-learn style accuracy scoring.

		Parameters:
		----------
		X_test : X_test
		y_test : y_test

		Returns:
		----------
		A scalar value of the accuracy score on the task.
		"""

		return np.sum(self.predict(X_test)==y_test) / y_test.size

	def get_column_names(self):
		""" Access column names."""

		if self.cols is None:
			raise NameError("Train model with pandas data frame to get column names.")

		return self.cols

	def fidelity(self, X):
		""" Get the fidelity of the adversarial model to the original predictions.  High fidelity means that
		we're predicting f along the in distribution data.
		
		Parameters:
		----------
		X : np.ndarray	

		Returns:
		----------
		The fidelity score of the adversarial model's predictions to the model you're trying to obscure's predictions.
		"""

		return (np.sum(self.predict(X) == self.f_obscure.predict(X)) / X.shape[0])

class Adversarial_Lime_Model(Adversarial_Model):
	""" Lime adversarial model.  Generates an adversarial model for LIME style explainers using the Adversarial Model
	base class.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	generator : String (either "Perturb" or "DropoutVAE")
	generator_specs : dictionary with data required by generator (Required if generator is DropoutVAE)
	"""
	def __init__(self, f_obscure, psi_display, generator = "Perturb", generator_specs={"perturbation_std": 0.3}):
		super(Adversarial_Lime_Model, self).__init__(f_obscure, psi_display)
		self.generator_specs = generator_specs

		if generator == "DropoutVAE":
			self.generator = DropoutVAE(original_dim = generator_specs["original_dim"],
                                        input_shape = (generator_specs["original_dim"],),
                                        intermediate_dim = generator_specs["intermediate_dim"],
                                        dropout = generator_specs["dropout"],
                                        latent_dim = generator_specs["latent_dim"])
			# MCD-VAE requires data in [0, 1]
			self.scaler = MinMaxScaler()

		elif generator == "Perturb":
			self.generator = None
			# Original perturbation sampling requires data with average 0 and variance 1
			self.scaler = StandardScaler()

		# Forest or RBF
		self.generator = generator

	def train(self, X, y, feature_names, perturbation_multiplier=30, categorical_features=[], integer_attributes=[], rf_estimators=100, estimator=None):
		""" Trains the adversarial LIME model.  This method trains the perturbation detection classifier to detect instances
		that are either in the manifold or not if no estimator is provided.
		
		Parameters:
		----------
		X : np.ndarray of pd.DataFrame
		y : np.ndarray
		perturbation_multiplier : int
		cols : list
		categorical_columns : list
		rf_estimators : integer
		estimaor : func
		"""
		if isinstance(X, pd.DataFrame):
			cols = [c for c in X]
			X = X.values
		elif not isinstance(X, np.ndarray):
			raise NameError("X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

		self.cols = feature_names
		all_x, all_y = [], []

		# Data normalization (not for treeEnsemble or rbfDataGen, data is just loaded there)
		if self.generator not in ["Forest", "RBF"]:
			X = self.scaler.fit_transform(X)

		# Generate samples with given data generator
		# Perturbations
		if self.generator is None:
			# loop over perturbation data to create larger data set
			for _ in range(perturbation_multiplier):
				perturbed_xtrain = np.random.normal(0,self.generator_specs["perturbation_std"], size=X.shape)
				p_train_x = np.vstack((X, X + perturbed_xtrain))
				p_train_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X.shape[0])))

				all_x.append(p_train_x)
				all_y.append(p_train_y)

			all_x = np.vstack(all_x)
			all_y = np.concatenate(all_y)

			# Reverse transofrmation back to the original dimensions
			all_x = self.scaler.inverse_transform(all_x)

		# treeEnsemble
		elif self.generator == "Forest":
			# load pregenerated data
			if self.generator_specs["experiment"] == "Compas":
				X_gen = pd.read_csv("../Data/compas_adversarial_train_forest.csv")
			elif self.generator_specs["experiment"] == "German":
				X_gen = pd.read_csv("../Data/german_adversarial_train_forest.csv")
			# CC dataset
			else:
				X_gen = pd.read_csv("../Data/cc_adversarial_train_forest.csv")
			# Create dummies (except for CC which does not have any categorical features)
			if self.generator_specs["experiment"] != "CC":
				X_gen = pd.get_dummies(X_gen)
				X_gen = X_gen[self.cols]
			all_x = np.concatenate((X, X_gen.values), axis = 0)
			all_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X_gen.shape[0])))

		# rbfDataGen
		elif self.generator == "RBF":
			# load pregenerated data
			if self.generator_specs["experiment"] == "Compas":
				X_gen = pd.read_csv("../Data/compas_adversarial_train_RBF.csv")
			elif self.generator_specs["experiment"] == "German":
				X_gen = pd.read_csv("../Data/german_adversarial_train_RBF.csv")
			# CC dataset
			else:
				X_gen = pd.read_csv("../Data/cc_adversarial_train_RBF.csv")
			# Create dummies (except for CC which does not have any categorical features)
			if self.generator_specs["experiment"] != "CC":
				X_gen = pd.get_dummies(X_gen)
				X_gen = X_gen[self.cols]
			all_x = np.concatenate((X, X_gen.values), axis = 0)
			all_y = np.concatenate((np.ones(X.shape[0]), np.zeros(X_gen.shape[0])))

		# MCD-VAE
		else:
			# Generator training
			X_train, X_val = train_test_split(X, test_size=0.5)
			self.generator.fit(X_train, X_val, epochs=self.generator_specs["epochs"], batch_size=50)

			# Generate new data
			X_val = np.reshape(X_val, (-1, X_train.shape[1]))
			encoded = self.generator.mean_predict(X_val, nums = 2*perturbation_multiplier)
			all_x = encoded.reshape(2*perturbation_multiplier*encoded.shape[2], X_train.shape[1])

			# Concatenate the original and sampled instances
			all_y = np.concatenate((np.zeros(all_x.shape[0]), np.ones(X_train.shape[0] + X_val.shape[0])))
			all_x = np.concatenate((all_x, X_train, X_val), axis=0)

			# Reverse transofrmation back to the original dimensions
			all_x = self.scaler.inverse_transform(all_x)

			# Round up the integer attributes
			all_x[:, integer_attributes] = (np.around(all_x[:, integer_attributes])).astype(int)

		# it's easier to just work with numerical columns, so focus on them for exploiting LIME
		self.numerical_cols = [feature_names.index(c) for c in feature_names if feature_names.index(c) not in categorical_features]

		if self.numerical_cols == []:
			raise NotImplementedError("We currently only support numerical column data. If your data set is all categorical, consider using SHAP adversarial model.")

		# generate perturbation detection model as RF
		xtrain = all_x[:,self.numerical_cols]

		xtrain, xtest, ytrain, ytest = train_test_split(xtrain, all_y, test_size=0.2)

		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain, ytrain)
		else:
			self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain, ytrain)

		ypred = self.perturbation_identifier.predict(xtest)
		self.ood_training_task_ability = (ytest, ypred)

		return self

class Adversarial_Kernel_SHAP_Model(Adversarial_Model):
	""" SHAP adversarial model.  Generates an adversarial model for SHAP style perturbations.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	"""
	def __init__(self, f_obscure, psi_display, **kwargs):
		super(Adversarial_Kernel_SHAP_Model, self).__init__(f_obscure, psi_display)
		self.generator = kwargs.get("generator", None)
		self.generator_specs = kwargs.get("generator_specs", None)
		# For MCD-VAE the scaler is added, because data needs to be scaled to [0, 1] interval
		if self.generator == "DropoutVAE":
			if self.generator_specs is None:
				raise ValueError("generator_specs should not be None")
			self.generator = DropoutVAE(original_dim = self.generator_specs["original_dim"],
                                        input_shape = (self.generator_specs["original_dim"],),
                                        intermediate_dim = self.generator_specs["intermediate_dim"],
                                        dropout = self.generator_specs["dropout"],
                                        latent_dim = self.generator_specs["latent_dim"])
			self.scaler = MinMaxScaler()

	def train(self, X, y, feature_names, background_distribution=None, perturbation_multiplier=10, n_samples=2e4, rf_estimators=100, n_kmeans=10, estimator=None,
				dummy_idcs = [], integer_idcs = []):
		""" Trains the adversarial SHAP model. This method perturbs the shap training distribution by sampling from 
		its kmeans (or distribution set generated by data generator) and randomly adding features.  These points get substituted into a test set.  We also check to make 
		sure that the instance isn't in the test set before adding it to the out of distribution set. If an estimator is 
		provided this is used.

		Parameters:
		----------
		X : np.ndarray
		y : np.ndarray
		features_names : list
		perturbation_multiplier : int
		n_samples : int or float
		rf_estimators : int
		n_kmeans : int
		estimator : func
		dummy_idcs : list of lists od indices, each list represents one categorical feature
		integer_idcs : list of indices of integer attributes that are not categorical

		Returns:
		----------
		The model itself.
		"""

		if isinstance(X, pd.DataFrame):
			X = X.values
		elif not isinstance(X, np.ndarray):
			raise NameError("X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

		self.cols = feature_names

		# Create the feature groups (dummy features describing the same categorical feature are in the same group)
		groups = list(range(X.shape[1]))
		for dummy in dummy_idcs:	
			for idx in dummy:
				groups.remove(idx)
			groups.append(np.array(dummy))
		# List that can be sorted
		groups = [np.array([el]) if not isinstance(el, np.ndarray) else el for el in groups]
		sortable = [el[0] for el in groups]
		groups = np.array(groups)
		# Sort the array
		groups = groups[np.argsort(sortable)]
		# Typecast back to list
		self.groups = groups.tolist()

		# If generator is MCD-VAE, it must be fit to train data
		if isinstance(self.generator, DropoutVAE):
			X_scaled = self.scaler.fit_transform(X)
			X_train, X_val = train_test_split(X_scaled, test_size = 0.5)
			self.generator.fit(X_train, X_val, epochs=self.generator_specs["epochs"], batch_size=50)

		# Create the mock background distribution we'll pull from to create substitutions
		# Perturbations
		if self.generator is None:
			if background_distribution is None:
				background_distribution = shap.kmeans(X,n_kmeans).data
		# In case of treeEnsemble and rbfDataGen we just load the whole distribution set
		elif self.generator in ["Forest", "RBF"]:
			if self.generator == "RBF":
				if self.generator_specs["experiment"] == "Compas":
					X_gen = pd.read_csv("../Data/compas_adversarial_train_RBF.csv")
				elif self.generator_specs["experiment"] == "German":
					X_gen = pd.read_csv("../Data/german_adversarial_train_RBF.csv")
				else:
					X_gen = pd.read_csv("../Data/cc_adversarial_train_RBF.csv")
			else:
				if self.generator_specs["experiment"] == "Compas":
					X_gen = pd.read_csv("../Data/compas_shap_adversarial_train_forest.csv")
				elif self.generator_specs["experiment"] == "German":
					X_gen = pd.read_csv("../Data/german_shap_adversarial_train_forest.csv")
				else:
					X_gen = pd.read_csv("../Data/cc_shap_adversarial_train_forest.csv")

			# Create dummies (except for CC which does not have any categorical features)
			if self.generator_specs["experiment"] != "CC":
				X_gen = pd.get_dummies(X_gen)
				X_gen = X_gen[feature_names]
			background_distribution = X_gen.values
		repeated_X = np.repeat(X, perturbation_multiplier, axis=0)

		new_instances = []
		equal = []

		# Generate sampled data
		for _ in range(int(n_samples)):
			i = np.random.choice(X.shape[0])
			point = deepcopy(X[i, :])

			# iterate over points, sampling and updating
			# We check if data generator is given (in that case we substitute feature values according to the groups)
			if self.generator is None:
				for _ in range(X.shape[1]):
					j = np.random.choice(X.shape[1])
					point[j] = deepcopy(background_distribution[np.random.choice(background_distribution.shape[0]),j])
			else:
				# With MCD-VAE we generate new distribution set for each instance
				if isinstance(self.generator, DropoutVAE):
					background_distribution = self.generate_data(point, dummy_idcs, integer_idcs, n_kmeans)
				for _ in range(len(self.groups)):
					j = np.random.choice(len(self.groups))
					point[self.groups[j]] = deepcopy(background_distribution[np.random.choice(background_distribution.shape[0]), self.groups[j]])
	
			new_instances.append(point)

		substituted_training_data = np.vstack(new_instances)
		all_instances_x = np.vstack((repeated_X, substituted_training_data))

		# make sure feature truly is out of distribution before labeling it
		xlist = X.tolist()
		ys = np.array([1 if substituted_training_data[val,:].tolist() in xlist else 0\
						 for val in range(substituted_training_data.shape[0])])

		all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]),ys))

		xtrain,xtest,ytrain,ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2)

		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain,ytrain)
		else:
			self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain,ytrain)

		ypred = self.perturbation_identifier.predict(xtest)
		self.ood_training_task_ability = (ytest, ypred)

		return self

	''' Generates the background distribution around instance x using MCD-VAE
	Parameters:
	x : 1D numpy array -> The point around which we generate the distribution
	dummy_idcs : list of lists od indeces, each list represents one categorical feature
	integer_idcs : list of indeces of integer attributes that are not categorical
	generated_size : int, number of generated instances
	'''
	def generate_data(self, x, dummy_idcs, integer_idcs, generated_size):
		reshaped = x.reshape(1, -1)
		scaled = self.scaler.transform(reshaped)
		scaled = np.reshape(scaled, (-1, reshaped.shape[1]))
		encoded = self.generator.mean_predict(scaled, nums = generated_size)

		generated_data = encoded.reshape(generated_size*encoded.shape[2], reshaped.shape[1])
		generated_data = self.scaler.inverse_transform(generated_data)

		# Round up the integer attributes
		generated_data[:, integer_idcs] = (np.around(generated_data[:, integer_idcs])).astype(int)

		# Correct values of dummy features to 0 and 1
		for feature in dummy_idcs:
			column = generated_data[:, feature]
			binary = np.zeros(column.shape)
			# We check if a feature has only two possible values, otherwise it has more dummies
			if len(feature) == 1:
				binary = (column > 0.5).astype(int)
			else:
				# Value 1 is given to the dummy with highest value
				ones = column.argmax(axis = 1)
				for i, idx in enumerate(ones):
					binary[i, idx] = 1
			generated_data[:, feature] = binary

		return generated_data

# This mdoel is not in original Fooling-LIME-SHAP repository
class Adversarial_IME_Model(Adversarial_Model):
	""" IME adversarial model.  Generates an adversarial model for IME style perturbations.

	Parameters:
	----------
	f_obscure : function
	psi_display : function
	"""
	def __init__(self, f_obscure, psi_display, **kwargs):
		super(Adversarial_IME_Model, self).__init__(f_obscure, psi_display)
		self.generator = kwargs.get("generator", None)
		self.generator_specs = kwargs.get("generator_specs", None)
		
		# For MCD-VAE the scaler is added, because data needs to be scaled to [0, 1] interval
		if self.generator == "DropoutVAE":
			if self.generator_specs is None:
				raise ValueError("generator_specs should not be None")
			self.generator = DropoutVAE(original_dim = self.generator_specs["original_dim"],
                                        input_shape = (self.generator_specs["original_dim"],),
                                        intermediate_dim = self.generator_specs["intermediate_dim"],
                                        dropout = self.generator_specs["dropout"],
                                        latent_dim = self.generator_specs["latent_dim"])
			self.scaler = MinMaxScaler()

	def train(self, X, y, feature_names, perturbation_multiplier=10, rf_estimators=100, estimator=None,
				dummy_idcs = [], integer_idcs = []):
		""" Trains the adversarial IME model. This method perturbs the training distribution by sampling from 
		it. For each instance we choose a random feature index and random permutation. We add one instance with
		features which indeces succeed the sampled one in the sampled permutation excluding the sampled feature
		changed to the sampled instances value. We add another instance including the sampled feature.We also check to make 
		sure that the instance isn't in the test set before adding it to the out of distribution set. If an estimator is 
		provided this is used.

		Parameters:
		----------
		X : np.ndarray
		y : np.ndarray
		features_names : list
		perturbation_multiplier : int
		rf_estimators : int
		estimator : func
		dummy_idcs : list of lists od indices, each list represents one categorical feature
		integer_idcs : list of indices of integer attributes that are not categorical

		Returns:
		----------
		The model itself.
		"""

		if isinstance(X, pd.DataFrame):
			X = X.values
		elif not isinstance(X, np.ndarray):
			raise NameError("X of type {} is not accepted. Only pandas dataframes or numpy arrays allowed".format(type(X)))

		self.cols = feature_names

		# If generator is MCD-VAE, it needs to be fitted to train data
		if isinstance(self.generator, DropoutVAE):
			X_scaled = self.scaler.fit_transform(X)
			X_train, X_val = train_test_split(X_scaled, test_size = 0.5)
			self.generator.fit(X_train, X_val, epochs=self.generator_specs["epochs"], batch_size=50)

		# Load generated data if generator is treeEnsemble with data fill option
		if self.generator == "Forest":
			if self.generator_specs["experiment"] == "Compas":
				X_gen = pd.read_csv("../Data/compas_ime_adversarial_train_forest.csv")
			elif self.generator_specs["experiment"] == "German":
				X_gen = pd.read_csv("../Data/german_ime_adversarial_train_forest.csv")
			else:
				X_gen = pd.read_csv("../Data/cc_ime_adversarial_train_forest.csv")

			# Create dummies (except for CC which does not have any categorical features)
			if self.generator_specs["experiment"] != "CC":
				X_gen = pd.get_dummies(X_gen)
				X_gen = X_gen[feature_names]
			generated_data = X_gen.values

		# We multiply the original instances as many times so their number is equal to the number of sampled instances
		repeated_X = np.repeat(X, 2*perturbation_multiplier, axis=0)

		new_instances = []
		equal = []

		# We generate perturbation_multiplier samples for each instance in training set
		for i in range(X.shape[0]):
			# We add perturbation_multiplier new instances
			for _ in range(perturbation_multiplier):
				# Sample new instance
				# If data generator is given, we generate/load instance w, otherwise it is randomly sampled from training set
				if self.generator is None:
					j = np.random.choice(X.shape[0])
					w = deepcopy(X[j, :])
				elif self.generator == "Forest":
					w = deepcopy(generated_data[i, :])
				else:
					w = self.generate_point(X[i,:], dummy_idcs, integer_idcs)
					# Reshape w to 1D array
					w = w.reshape((-1,))

				# We sample random feature index
				feature_idx = np.random.choice(X.shape[1])
				# Random permutation
				ids = np.arange(X.shape[1])
				np.random.shuffle(ids)
				# Position of sampled feature in the permutation
				pos = np.where(ids == feature_idx)[0][0]
				# Point with excluded sampled feature
				new_point1 = deepcopy(X[i, :])
				new_point1[ids[pos+1:]] = w[ids[pos+1:]]
				new_instances.append(new_point1)
				# Point with included sampled feature
				new_point2 = deepcopy(X[i, :])
				new_point2[ids[pos:]] = w[ids[pos:]]
				new_instances.append(new_point2)

		substituted_training_data = np.vstack(new_instances)
		all_instances_x = np.vstack((repeated_X, substituted_training_data))

		# make sure feature truly is out of distribution before labeling it
		xlist = X.tolist()
		ys = np.array([1 if substituted_training_data[val,:].tolist() in xlist else 0\
						 for val in range(substituted_training_data.shape[0])])

		all_instances_y = np.concatenate((np.ones(repeated_X.shape[0]),ys))

		xtrain,xtest,ytrain,ytest = train_test_split(all_instances_x, all_instances_y, test_size=0.2)

		if estimator is not None:
			self.perturbation_identifier = estimator.fit(xtrain,ytrain)
		else:
			self.perturbation_identifier = RandomForestClassifier(n_estimators=rf_estimators).fit(xtrain,ytrain)

		ypred = self.perturbation_identifier.predict(xtest)
		self.ood_training_task_ability = (ytest, ypred)

		return self

	'''
	Function that generates new point using MCD-VAE.

	Parameters:
		----------
		data_row : Row of np.ndarray
		estimator : func
		dummy_idcs : list of lists od indeces, each list represents one categorical feature
		integer_idcs : list of indeces of integer attributes that are not categorical

		Returns:
		----------
		np.ndarray : generated data point
	'''
	def generate_point(self, data_row, dummy_idcs, integer_idcs):
		# So we have array of shape (1, X.shape[1])
		reshaped = data_row.reshape(1, -1)
		scaled = self.scaler.transform(reshaped)
		scaled = np.reshape(scaled, (-1, reshaped.shape[1]))
		encoded = self.generator.mean_predict(scaled, nums = 1)

		generated_data = encoded.reshape(encoded.shape[2], reshaped.shape[1])

		# Rescaling back to original space and rounding of integer features
		generated_data = self.scaler.inverse_transform(generated_data)
		generated_data[:, integer_idcs] = (np.around(generated_data[:, integer_idcs])).astype(int)

		# Set the values of dummy features to 0 or 1
		for feature in dummy_idcs:
			column = generated_data[:, feature]
			binary = np.zeros(column.shape)
			# We check if a feature has only two possible values, otherwise it has more dummies
			if len(feature) == 1:
				binary = (column > 0.5).astype(int)
			else:
				# Value 1 is given to the dummy with highest value
				ones = column.argmax(axis = 1)
				for i, idx in enumerate(ones):
					binary[i, idx] = 1
			generated_data[:, feature] = binary

		return generated_data