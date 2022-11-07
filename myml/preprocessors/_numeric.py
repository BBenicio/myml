"""
Data preprocessors for numeric features.

The `preprocessors` list contains scikit-learn compatible transformers
that handle numeric features.
"""

import sklearn.preprocessing


preprocessors = [
    sklearn.preprocessing.FunctionTransformer(),
    sklearn.preprocessing.MaxAbsScaler(),
    sklearn.preprocessing.MinMaxScaler(),
    sklearn.preprocessing.StandardScaler(),
    sklearn.preprocessing.PowerTransformer(),
    sklearn.preprocessing.RobustScaler()
]
""" List of transformers to deal with numeric features. """
