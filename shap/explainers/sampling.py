from ..common import convert_to_instance, convert_to_model, match_instance_to_data, match_model_to_data, convert_to_instance_with_index, convert_to_link, IdentityLink, convert_to_data, DenseData
from .kernel import KernelExplainer
import numpy as np
import pandas as pd
import logging
import math

from scipy.stats import norm

log = logging.getLogger('shap')

from Generators.DropoutVAE import DropoutVAE

from copy import deepcopy

import time

class SamplingExplainer(KernelExplainer):
    """ This is an extension of the Shapley sampling values explanation method (aka. IME)

    SamplingExplainer computes SHAP values under the assumption of feature independence and is an
    extension of the algorithm proposed in "An Efficient Explanation of Individual Classifications
    using Game Theory", Erik Strumbelj, Igor Kononenko, JMLR 2010. It is a good alternative to
    KernelExplainer when you want to use a large background set (as opposed to a single reference
    value for example).

    Parameters
    ----------
    model : function
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes a the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array or pandas.DataFrame
        The background dataset to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero. Unlike the
        KernelExplainer this data can be the whole training set, even if that is a large set. This
        is because SamplingExplainer only samples from this background dataset.
    """

    def __init__(self, model, data, **kwargs):
        # silence warning about large datasets
        level = log.level
        log.setLevel(logging.ERROR)
        super(SamplingExplainer, self).__init__(model, data, **kwargs)
        log.setLevel(level)

        assert str(self.link) == "identity", "SamplingExplainer only supports the identity link not " + str(self.link)

    '''
    Additional arguments that are not included in original SHAP package (located in kwargs):
        nsamples: If set to "variance", then the number of samples is determined by the variance ofthe population,
            like it is described in the paper. If set to "auto" or not given, the number of samples is determined
            as in original SHAP package.
        alpha: float in [0, 1], the probability that error is bigger than desired, as in paper. Default value: 0.99
        expected_error: float, desired error size, as in paper. Default value: 0.01
        is_experiment: Boolean, if True, variance of popuation, execution time and error of the estimates are saved.
            Used for experiment of IME convergence rate. False by default
    '''
    def explain(self, incoming_instance, **kwargs):
        is_experiment = kwargs.get("is_experiment", False)

        if is_experiment:
            start_time = time.time()

        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        match_instance_to_data(instance, self.data)

        # In case of MCD-VAE we generate the neighbourhood set of incoming_instance to take samples from
        if (isinstance(self.generator, DropoutVAE)):
            self.generated_data = self.generate_around_point(incoming_instance)

        if self.generator == "Forest":
            # Next distribution set
            self.distribution_set = self.forest_data[self.forest_index:self.forest_index + self.distribution_size, :]
            # Update the index
            self.forest_index += self.distribution_size

            self.data = convert_to_data(self.distribution_set, keep_index=self.keep_index)

        assert len(self.data.groups) == self.P, "SamplingExplainer does not support feature groups!"

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = self.varying_groups(instance.x)
        #self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
        self.M = len(self.varyingInds)

        # find f(x)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values[0]
        self.fx = model_out[0]

        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then there no feature has an effect
        if self.M == 0:
            phi = np.zeros((len(self.data.groups), self.D))
            phi_var = np.zeros((len(self.data.groups), self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((len(self.data.groups), self.D))
            phi_var = np.zeros((len(self.data.groups), self.D))
            diff = self.fx - self.fnull
            for d in range(self.D):
                phi[self.varyingInds[0],d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:

            # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples in ["auto", "variance"]:
                self.nsamples = 1000 * self.M
            assert self.nsamples % 2 == 0, "nsamples must be divisible by 2!"

            min_samples_per_feature = kwargs.get("min_samples_per_feature", 100)
            round1_samples = self.nsamples
            round2_samples = 0
            if round1_samples > self.M * min_samples_per_feature:
                round2_samples = round1_samples - self.M * min_samples_per_feature
                round1_samples -= round2_samples

            # divide up the samples among the features for round 1
            nsamples_each1 = np.ones(self.M, dtype=np.int64) * 2 * (round1_samples // (self.M * 2))
            for i in range((round1_samples % (self.M * 2)) // 2):
                nsamples_each1[i] += 2

            # explain every feature in round 1
            phi = np.zeros((self.P, self.D))
            phi_var = np.zeros((self.P, self.D))
            self.X_masked = np.zeros((nsamples_each1.max(), self.data.data.shape[1]))
            for i,ind in enumerate(self.varyingInds):
                phi[ind,:],phi_var[ind,:] = self.sampling_estimate(ind, self.model.f, instance.x, self.data.data, nsamples=nsamples_each1[i])

            if (self.nsamples ==  "variance"):
                # we determine the number of samples for round 2 based on variances from round 1
                if (self.D == 1):
                    table_var = deepcopy(np.reshape(phi_var, (-1,)))
                    alpha = kwargs.get("alpha", 0.99)
                    expected_error = kwargs.get("expected_error", 0.01)
                    z = norm.ppf((1+alpha)/2)
                    self.nsamples = math.ceil((z**2)*(np.sum(table_var)) / (expected_error**2))
                    if round1_samples > self.nsamples:
                        round2_samples = 0
                    else:
                        round2_samples = self.nsamples - round1_samples

                else:
                    raise ValueError("Choosing number of samples according to variance implemented only for vector output")

            # optimally allocate samples according to the variance
            if phi_var.sum() == 0:
                phi_var += 1 # spread samples uniformally if we found no variability
            phi_var /= phi_var.sum()
            nsamples_each2 = (phi_var[self.varyingInds,:].mean(1) * round2_samples).astype(np.int)
            for i in range(len(nsamples_each2)):
                if nsamples_each2[i] % 2 == 1: nsamples_each2[i] += 1
            for i in range(len(nsamples_each2)):
                if nsamples_each2.sum() > round2_samples:
                    nsamples_each2[i] -= 2
                elif nsamples_each2.sum() < round2_samples:
                    nsamples_each2[i] += 2
                else:
                    break

            self.X_masked = np.zeros((nsamples_each2.max(), self.data.data.shape[1]))
            for i,ind in enumerate(self.varyingInds):
                if nsamples_each2[i] > 0:
                    val,var = self.sampling_estimate(ind, self.model.f, instance.x, self.data.data, nsamples=nsamples_each2[i])

                    total_samples = nsamples_each1[i] + nsamples_each2[i]
                    phi[ind,:] = (phi[ind,:] * nsamples_each1[i] + val * nsamples_each2[i]) / total_samples
                    phi_var[ind,:] = (phi_var[ind,:] * nsamples_each1[i] + var * nsamples_each2[i]) / total_samples

            # convert from the variance of the differences to the variance of the mean (phi)
            for i,ind in enumerate(self.varyingInds):
                phi_var[ind,:] /= np.sqrt(nsamples_each1[i] + nsamples_each2[i])

            # correct the sum of the SHAP values to equal the output of the model using a linear
            # regression model with priors of the coefficents equal to the estimated variances for each
            # SHAP value (note that 1e6 is designed to increase the weight of the sample and so closely
            # match the correct sum)
            sum_error = self.fx - phi.sum(0) - self.fnull
            for i in range(self.D):
                # this is a ridge regression with one sample of all ones with sum_error[i] as the label
                # and 1/v as the ridge penalties. This simlified (and stable) form comes from the
                # Sherman-Morrison formula
                v = (phi_var[:,i] / phi_var[:,i].max()) * 1e6
                adj = sum_error[i] * (v - (v * v.sum()) / (1 + v.sum()))
                phi[:,i] += adj

        if phi.shape[1] == 1:
            phi = phi[:,0]

        # Add the data to the table if experiment is being conducted
        if is_experiment:
            end_time = time.time()
            if self.D == 1:
                table_phi = deepcopy(np.reshape(phi, (-1,)))
            else:
                raise ValueError("Varaince experiment implemented only for vector output.")
            self.update_experiment_table(table_var, table_phi, end_time - start_time)

        return phi

    def sampling_estimate(self, j, f, x, X, nsamples=10, generated_data=None):
        assert nsamples % 2 == 0, "nsamples must be divisible by 2!"
        X_masked = self.X_masked[:nsamples,:]
        inds = np.arange(X.shape[1])

        # Basic perturbations
        if self.generator is None:
            for i in range(0, nsamples//2):
                np.random.shuffle(inds)
                pos = np.where(inds == j)[0][0]
                rind = np.random.randint(X.shape[0])
                X_masked[i, :] = x
                X_masked[i, inds[pos+1:]] = X[rind, inds[pos+1:]]
                X_masked[-(i+1), :] = x
                X_masked[-(i+1), inds[pos:]] = X[rind, inds[pos:]]

        # Forest with data fill
        elif self.generator == "Forest":
            for i in range(0, nsamples//2):
                np.random.shuffle(inds)
                pos = np.where(inds == j)[0][0]
                rind = np.random.randint(self.distribution_set.shape[0])
                X_masked[i, :] = x
                X_masked[i, inds[pos+1:]] = self.distribution_set[rind, inds[pos+1:]]
                X_masked[-(i+1), :] = x
                X_masked[-(i+1), inds[pos:]] = self.distribution_set[rind, inds[pos:]]

        # DropoutVAE
        else:
            for i in range(0, nsamples//2):
                np.random.shuffle(inds)
                pos = np.where(inds == j)[0][0]
                rind = np.random.randint(self.generated_data.shape[0])
                X_masked[i, :] = x
                X_masked[i, inds[pos+1:]] = self.generated_data[rind, inds[pos+1:]]
                X_masked[-(i+1), :] = x
                X_masked[-(i+1), inds[pos:]] = self.generated_data[rind, inds[pos:]]

        evals = f(X_masked)
        evals_on = evals[:nsamples//2]
        evals_off = evals[nsamples//2:][::-1]
        d = evals_on - evals_off

        return np.mean(d, 0), np.var(d, 0)

    '''
    Function that generates new point using DropoutVAE.

    Parameters:
    ----------
    data_row : Row of np.ndarray

    Returns:
    ----------
    np.ndarray : generated data point
    '''
    def generate_around_point(self, data_row):
        # So we have array of shape (1, X.shape[1])
        reshaped = data_row.reshape(1, -1)
        scaled = self.scaler.transform(reshaped)
        scaled = np.reshape(scaled, (-1, reshaped.shape[1]))
        encoded = self.generator.mean_predict(scaled, nums = self.instance_multiplier)

        generated_data = encoded.reshape(self.instance_multiplier*encoded.shape[2], reshaped.shape[1])

        # Rescaling of the data back to original space and rounding of the integer features
        generated_data = self.scaler.inverse_transform(generated_data)
        generated_data[:, self.integer_idcs] = (np.around(generated_data[:, self.integer_idcs])).astype(int)

        # Correct the dummy values so we have one-hot encoding
        for feature in self.dummy_idcs:
            column = generated_data[:, feature]
            binary = np.zeros(column.shape)
            # Check whether the feature has only 2 possible values, otherwise it has more than one dummy
            if len(feature) == 1:
                binary = (column > 0.5).astype(int)
            else:
                # Set the dummy variable with highest value to 1
                ones = column.argmax(axis = 1)
                for i, idx in enumerate(ones):
                    binary[i, idx] = 1
            generated_data[:, feature] = binary

        return generated_data

    '''
    Function that creates the empty list which will containt data for dataframe returned by get_experiment_dataframe.
    Used by experiment for IME convergence rate. Should be called before the experiment. User has to provide true
    Shapley values.

    Parameters:
    ----------
    real_phi : np.ndarray containing real shapley values for each example we are explaining
    '''
    def create_experiment_table(self, real_phi):
        self.real_phi = real_phi

        self.experiment_table = []

    '''
    Function that appends the data for current example to the table. Used by experiment for IME convergence rate.

    Parameters:
    ----------
    var : 1D numpy array containing variance of population for each feature
    phi : 1D numpy array containing feature contributions obtained by IME
    time : number, time of execution of IME on current example
    '''
    def update_experiment_table(self, var, phi, time):
        # Data row we will append
        new_row = []

        # Insert variances
        for i in range(len(var)):
            new_row.append(var[i])
        # Insert average variance
        avg_var = np.sum(var) / len(var)
        new_row.append(avg_var)
        # Insert numver of samples
        new_row.append(self.nsamples)
        # Insert phi errors
        shapley_values = self.real_phi[len(self.experiment_table), :]
        delta_phi = shapley_values - phi
        for i in range(len(delta_phi)):
            new_row.append(delta_phi[i])
        # Insert mean absolute error
        new_row.append(np.sum(abs(delta_phi)) / len(delta_phi))
        # Insert time
        new_row.append(time)

        self.experiment_table.append(new_row)

    '''
    Function that returns dataframe with results of the IME convergence rate experiment
    '''
    def get_experiment_dataframe(self):
        n = (len(self.experiment_table[0]) - 4) // 2

        # List of coumn names for the table
        column_names = []
        # First we add variances
        for i in range(n):
            column_names.append("var_"+str(i+1))
        # We add average variance
        column_names.append("avg_var")
        # Number of samples
        column_names.append("n_samples")
        # We add phi_errors
        for i in range(n):
            column_names.append("delta_phi_"+str(i+1))
        # We add mean absolute error of phi
        column_names.append("error")
        # We add time needed to explain
        column_names.append("time")

        return pd.DataFrame(self.experiment_table, columns = column_names)