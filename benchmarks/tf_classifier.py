'''
seed:843484287
digits:0.953704
seed:193953994
Climate Model Crashes:0.938272
'''
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata
import numpy as np

import sys
if sys.version_info.major == 2:
    # Python 2
    from urllib2 import HTTPError
    from urllib import urlopen
else:
    from urllib.error import HTTPError
    from urllib import urlopen

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

DATASETS = [
    "digits", 
    "Climate Model Crashes"
    ]
MODELS = [
    tf.estimator.LinearClassifier, 
    #tf.estimator.DNNClassifier,
    #tf.estimator.DNNLinearCombinedClassifier,
    ]
for dataset in DATASETS:
	for model in MODELS:
		import os;os.system("rm -rf /tmp/tmpNIBUY")
		X,y = load_data_target(dataset)
		seed=np.random.randint(0, 2**30, 1)[0]
		#print "seed:%s" % seed
		X_train, y_train, X_test, y_test = split_normalize(X, y, random_state=seed)

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
		estimator = model(feature_columns=linear_feature_columns,
		                                   model_dir="/tmp/tmpNIBUY",
		                                   n_classes=len(np.unique(y_train)),
		                                   )

		estimator.train(input_fn=train_input_fn, steps=200)

		accuracy_score = estimator.evaluate(input_fn=test_input_fn)["accuracy"]
		print "%s:%s" % (dataset, accuracy_score)