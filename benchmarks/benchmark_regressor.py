"""

"""
from collections import defaultdict
from datetime import datetime
import json
import sys
import math

if sys.version_info.major == 2:
    # Python 2
    from urllib2 import HTTPError
    from urllib import urlopen
else:
    from urllib.error import HTTPError
    from urllib import urlopen

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.externals.joblib import delayed
from sklearn.externals.joblib import Parallel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss

from estimator_optimize.optimizer import gbrt_minimize
from estimator_optimize.optimizer import gp_minimize
from estimator_optimize.optimizer import forest_minimize
from estimator_optimize.space import Categorical
from estimator_optimize.space import Integer
from estimator_optimize.space import Real
from estimator_optimize.linear import LinearRegressor
from estimator_optimize.dnn import DNNRegressor
from estimator_optimize.dnn_linear_combined import DNNLinearCombinedRegressor

import tensorflow as tf
from tensorflow.python.ops import nn

# functions below are used to apply non - linear maps to parameter values, eg
# -3.0 -> 0.001
def pow10map(x):
    return 10.0 ** x

def pow2intmap(x):
    return int(2.0 ** x)

def nop(x):
    return x

linearparams = {
    #'linear.optimizer': (Categorical(['Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD']), nop),
    'linear.optimizer': (Categorical(['Adagrad', 'Adam', 'SGD']), nop),
    'linear.learning_rate': (Real(-5.0, -1), pow10map),
    'linear.initial_accumulator_value': (Real(0.01, 0.1), nop),
    'linear.beta1': (Real(0.1, 0.98), nop),
    'linear.beta2': (Real(0.1, 0.9999999), nop),
    'linear.epsilon': (Real(-10.0, -5.0), pow10map),
    'linear.learning_rate_power': (Real(-0.5, -0.0), nop),
    'linear.l1_regularization_strength': (Real(0.0, 0.1), nop),
    'linear.l2_regularization_strength': (Real(0.0, 0.1), nop),
    'linear.l2_shrinkage_regularization_strength': (Real(0.0, 0.1), nop),
    'linear.decay': (Real(0.1, 0.9), nop),
    'linear.momentum': (Real(0.0, 0.1), nop),
    'steps':(Real(5.0, 8.0), pow2intmap),
}

dnnparams = {
    'dnn.optimizer': (Categorical(['Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD']), nop),
    'dnn.learning_rate': (Real(-5.0, -1), pow10map),
    'dnn.initial_accumulator_value': (Real(0.01, 0.1), nop),
    'dnn.beta1': (Real(0.1, 0.98), nop),
    'dnn.beta2': (Real(0.1, 0.9999999), nop),
    'dnn.epsilon': (Real(-10.0, -5.0), pow10map),
    'dnn.learning_rate_power': (Real(-0.5, 0.0), nop),
    'dnn.l1_regularization_strength': (Real(0.0, 0.1), nop),
    'dnn.l2_regularization_strength': (Real(0.0, 0.1), nop),
    'dnn.l2_shrinkage_regularization_strength': (Real(0.0, 0.1), nop),
    'dnn.decay': (Real(0.1, 0.9), nop),
    'dnn.momentum': (Real(0.0, 0.1), nop),
    'activation_fn': (Categorical([nn.relu, nn.tanh, nn.elu, nn.sigmoid]), nop),
    'dropout': (Real(0.5, 0.9), nop),
    'steps':(Real(5.0, 8.0), pow2intmap),
    #'hidden_units':(Categorical([[1024, 512, 256], [512, 256, 128]]), nop),
}

dlparams = {
    'linear.optimizer': (Categorical(['Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD']), nop),
    'linear.learning_rate': (Real(-5.0, -1), pow10map),
    'linear.initial_accumulator_value': (Real(0.01, 0.1), nop),
    'linear.beta1': (Real(0.1, 0.98), nop),
    'linear.beta2': (Real(0.1, 0.9999999), nop),
    'linear.epsilon': (Real(-10.0, -5.0), pow10map),
    'linear.learning_rate_power': (Real(-0.5, -0.0), nop),
    'linear.l1_regularization_strength': (Real(0.0, 0.1), nop),
    'linear.l2_regularization_strength': (Real(0.0, 0.1), nop),
    'linear.l2_shrinkage_regularization_strength': (Real(0.0, 0.1), nop),
    'linear.decay': (Real(0.1, 0.9), nop),
    'linear.momentum': (Real(0.0, 0.1), nop),
    'dnn.optimizer': (Categorical(['Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD']), nop),
    'dnn.learning_rate': (Real(-5.0, -1), pow10map),
    'dnn.initial_accumulator_value': (Real(0.01, 0.1), nop),
    'dnn.beta1': (Real(0.1, 0.98), nop),
    'dnn.beta2': (Real(0.1, 0.9999999), nop),
    'dnn.epsilon': (Real(-10.0, -5.0), pow10map),
    'dnn.learning_rate_power': (Real(-0.5, 0.0), nop),
    'dnn.l1_regularization_strength': (Real(0.0, 0.1), nop),
    'dnn.l2_regularization_strength': (Real(0.0, 0.1), nop),
    'dnn.l2_shrinkage_regularization_strength': (Real(0.0, 0.1), nop),
    'dnn.decay': (Real(0.1, 0.9), nop),
    'dnn.momentum': (Real(0.0, 0.1), nop),
    'dnn_activation_fn': (Categorical([nn.relu, nn.tanh, nn.elu, nn.sigmoid]), nop),
    'dnn_dropout': (Real(0.5, 0.9), nop),
    'steps':(Real(5.0, 8.0), pow2intmap),
    #'hidden_units':(Categorical([[1024, 512, 256], [512, 256, 128]]), nop),
}

MODELS = {
    #LinearRegressor: linearparams,
    #DNNRegressor: dnnparams,
    DNNLinearCombinedRegressor: dlparams,
}

# every dataset should have have a mapping to the mixin that can handle it.
DATASETS = ["Boston", "Housing"]

# bunch of dataset preprocessing functions below
def split_normalize(X, y, random_state):
    """
    Splits data into training and validation parts.
    Test data is assumed to be used after optimization.

    Parameters
    ----------
    * `X` [array-like, shape = (n_samples, n_features)]:
        Training data.

    * `y`: [array-like, shape = (n_samples)]:
        Target values.

    Returns
    -------
    Split of data into training and validation sets.
    70% of data is used for training, rest for validation.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=random_state)
    sc = StandardScaler()
    sc.fit(X_train, y_train)
    X_train, X_test = sc.transform(X_train), sc.transform(X_test)
    return X_train, y_train, X_test, y_test


# this is used to process the output of fetch_mldata
def load_data_target(name):
    """
    Loads data and target given the name of the dataset.
    """
    if name == "Boston":
        data = load_boston()
    elif name == "Housing":
        data = fetch_california_housing()
        dataset_size = 1000 # this is necessary so that SVR does not slow down too much
        data["data"] = data["data"][:dataset_size]
        data["target"] =data["target"][:dataset_size]
    elif name == "digits":
        data = load_digits()
    elif name == "Climate Model Crashes":
        try:
            data = fetch_mldata("climate-model-simulation-crashes")
        except HTTPError as e:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00252/pop_failures.dat"
            data = urlopen(url).read().split('\n')[1:]
            data = [[float(v) for v in d.split()] for d in data]
            samples = np.array(data)
            data = dict()
            data["data"] = samples[:, :-1]
            data["target"] = np.array(samples[:, -1], dtype=np.int)
    else:
        raise ValueError("dataset not supported.")
    return data["data"], data["target"]


class MLBench(object):
    """
    A class which is used to perform benchmarking of black box optimization
    algorithms on various machine learning problems.
    On instantiation, the dataset is loaded that is used for experimentation
    and is kept in memory in order to avoid delays due to reloading of data.

    Parameters
    ----------
    * `model`: scikit-learn estimator
        An instance of a sklearn estimator.

    * `dataset`: str
        Name of the dataset.

    * `random_state`: seed
        Initialization for the random number generator in numpy.
    """
    def __init__(self, model, dataset, random_state):
        X, Y = load_data_target(dataset)
        self.X_train, self.y_train, self.X_test, self.y_test = split_normalize(
            X, Y, random_state)
        self.random_state = random_state
        self.model = model
        self.space = MODELS[model]


    def evaluate(self, point):
        """
        Fits model using the particular setting of hyperparameters and
        evaluates the model validation data.

        Parameters
        ----------
        * `point`: dict
            A mapping of parameter names to the corresponding values

        Returns
        -------
        * `score`: float
            Score (more is better!) for some specific point
        """
        X_train, y_train, X_test, y_test = (
            self.X_train, self.y_train, self.X_test, self.y_test)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(X_train)},
            y=np.array(y_train),
            num_epochs=None,
            shuffle=True)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(X_test)},
            y=np.array(y_test),
            num_epochs=1,
            shuffle=False)
        shape = [X_train.shape[1]]
        linear_feature_columns = [tf.feature_column.numeric_column("x", shape=shape)]

        # apply transformation to model parameters, for example exp transformation
        point_mapped = {}
        dnn_optimizer = {}
        linear_optimizer = {}
        for param, val in point.items():
            if param[0:4] == "dnn.":
                dnn_optimizer[param[4:]] = self.space[param][1](val)
            elif param[0:7] == "linear.":
                linear_optimizer[param[7:]] = self.space[param][1](val)
            else:
                point_mapped[param] = self.space[param][1](val)

        import os;os.system("rm -rf /tmp/tmpNIBUY")
        if self.model == LinearRegressor:
            estimator = self.model(feature_columns=linear_feature_columns,
                                   model_dir="/tmp/tmpNIBUY",
                                   optimizer=linear_optimizer,
                                   )
        elif self.model == DNNRegressor:
            estimator = self.model(hidden_units=[1024, 512, 256],
                                   feature_columns=linear_feature_columns,
                                   model_dir="/tmp/tmpNIBUY",
                                   optimizer=dnn_optimizer,
                                   activation_fn=point_mapped['activation_fn'],
                                   dropout=point_mapped['dropout']
                                   )
        elif self.model == DNNLinearCombinedRegressor:
            estimator = self.model(linear_feature_columns=linear_feature_columns,
                                   model_dir="/tmp/tmpNIBUY",
                                   dnn_feature_columns=[],
                                   linear_optimizer=linear_optimizer,
                                   dnn_optimizer=dnn_optimizer,
                                   dnn_activation_fn=point_mapped['dnn_activation_fn'],
                                   dnn_dropout=point_mapped['dnn_dropout']
                                   )
        max_loss = 5000.0
        try:
            estimator.train(input_fn=train_input_fn, steps=point_mapped['steps'])

            average_loss = estimator.evaluate(input_fn=test_input_fn)["average_loss"]
            print("average_loss:%s" % average_loss)
            if math.isnan(average_loss):
                average_loss = max_loss
            accuracy_score = min(accuracy_score, max_loss) 
        except BaseException as ex:
            print(ex)
            average_loss = max_loss

        return average_loss

# this is necessary to generate table for README in the end
table_template = """|Blackbox Function| Minimum | Best minimum | Mean f_calls to min | Std f_calls to min | Fastest f_calls to min
------------------|------------|-----------|---------------------|--------------------|-----------------------
| """

def calculate_performance(all_data):
    """
    Calculates the performance metrics as found in "benchmarks" folder of
    scikit-optimize and prints them in console.

    Parameters
    ----------
    * `all_data`: dict
        Traces data collected during run of algorithms. For more details, see
        'evaluate_optimizer' function.
    """

    sorted_traces = defaultdict(list)

    for model in all_data:
        for dataset in all_data[model]:
            for algorithm in all_data[model][dataset]:
                data = all_data[model][dataset][algorithm]

                # leave only best objective values at particular iteration
                best = [[v[-1] for v in d] for d in data]

                supervised_learning_type = "Regression" if ("Regressor" in model) else "Classification"

                # for every item in sorted_traces it is 2d array, where first dimension corresponds to
                # particular repeat of experiment, and second dimension corresponds to index
                # of optimization step during optimization
                key = (algorithm, supervised_learning_type)
                sorted_traces[key].append(best)

    # calculate averages
    for key in sorted_traces:
        # the meta objective: average over multiple tasks
        mean_obj_vals = np.mean(sorted_traces[key], axis=0)

        minimums = np.min(mean_obj_vals, axis=1)
        f_calls = np.argmin(mean_obj_vals, axis=1)

        min_mean = np.mean(minimums)
        min_stdd = np.std(minimums)
        min_best = np.min(minimums)

        f_mean = np.mean(f_calls)
        f_stdd = np.std(f_calls)
        f_best = np.min(f_calls)

        def fmt(float_value):
            return ("%.3f" % float_value)

        output = str(key[0]) + " | " + " | ".join(
            [fmt(min_mean) + " +/- " + fmt(min_stdd)] + [fmt(v) for v in [min_best, f_mean, f_stdd, f_best]])
        result = table_template + output
        print("")
        print(key[1])
        print(result)


def evaluate_optimizer(surrogate_minimize, model, dataset, n_calls, random_state):
    """
    Evaluates some estimator for the task of optimization of parameters of some
    model, given limited number of model evaluations.

    Parameters
    ----------
    * `surrogate_minimize`:
        Minimization function from skopt (eg gp_minimize) that is used
        to minimize the objective.
    * `model`: scikit-learn estimator.
        sklearn estimator used for parameter tuning.
    * `dataset`: str
        Name of dataset to train ML model on.
    * `n_calls`: int
        Budget of evaluations
    * `random_state`: seed
        Set the random number generator in numpy.

    Returns
    -------
    * `trace`: list of tuples
        (p, f(p), best), where p is a dictionary of the form "param name":value,
        and f(p) is performance achieved by the model for configuration p
        and best is the best value till that index.
        Such a list contains history of execution of optimization.
    """
    # below seed is necessary for processes which fork at the same time
    # so that random numbers generated in processes are different
    np.random.seed(random_state)
    problem = MLBench(model, dataset, random_state)
    space = problem.space
    dimensions_names = sorted(space)
    dimensions = [space[d][0] for d in dimensions_names]

    def objective(x):
        # convert list of dimension values to dictionary
        x = dict(zip(dimensions_names, x))
        # the result of "evaluate" is average_loss, which is the less the better
        y = problem.evaluate(x)
        return y

    # optimization loop
    result = surrogate_minimize(objective, dimensions, n_calls=n_calls, random_state=random_state)
    trace = []
    min_y = np.inf
    for x, y in zip(result.x_iters, result.func_vals):
        min_y = min(y, min_y)
        x_dct = dict(zip(dimensions_names, x))
        trace.append((x_dct, y, min_y))

    print(random_state)
    return trace


def run(n_calls=32, n_runs=1, save_traces=True, n_jobs=1):
    """
    Main function used to run the experiments.

    Parameters
    ----------
    * `n_calls`: int
        Evaluation budget.

    * `n_runs`: int
        Number of times to repeat the optimization in order to average out noise.

    * `save_traces`: bool
        Whether or not to save data collected during optimization

    * `n_jobs`: int
        Number of different repeats of optimization to run in parallel.
    """
    surrogate_minimizers = [gbrt_minimize, forest_minimize, gp_minimize]
    selected_models = sorted(MODELS, key=lambda x: x.__name__)

    # all the parameter values and objectives collected during execution are stored in list below
    all_data = {}
    for model in selected_models:
        all_data[model] = {}

        for dataset in DATASETS:

            all_data[model][dataset] = {}
            for surrogate_minimizer in surrogate_minimizers:
                print(surrogate_minimizer.__name__, model.__name__, dataset)
                seeds = np.random.randint(0, 2**30, n_runs)
                raw_trace = Parallel(n_jobs=n_jobs)(
                    delayed(evaluate_optimizer)(
                        surrogate_minimizer, model, dataset, n_calls, seed
                    ) for seed in seeds
                )
                all_data[model][dataset][surrogate_minimizer.__name__] = raw_trace

    # convert the model keys to strings so that results can be saved as json
    all_data = {k.__name__: v for k,v in all_data.items()}

    # dump the recorded objective values as json
    if save_traces:
        with open(datetime.now().strftime("%m_%Y_%d_%H_%m_%s")+'.json', 'w') as f:
            json.dump(all_data, f)
    calculate_performance(all_data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_calls', nargs="?", default=50, type=int,
        help="Number of function calls.")
    parser.add_argument(
        '--n_runs', nargs="?", default=10, type=int,
        help="Number of re-runs of single algorithm on single instance of a "
        "problem, in order to average out the noise.")
    parser.add_argument(
        '--save_traces', nargs="?", default=False, type=bool,
        help="Whether to save pairs (point, objective, best_objective) obtained"
        " during experiments in a json file.")
    parser.add_argument(
        '--n_jobs', nargs="?", default=1, type=int,
        help="Number of worker processes used for the benchmark.")

    args = parser.parse_args()
    run(args.n_calls, args.n_runs, args.save_traces, args.n_jobs)
