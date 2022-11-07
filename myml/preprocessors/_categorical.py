"""
Data preprocessors for categorical features.

The `preprocessors` list contains scikit-learn compatible transformers
that handle categorical features.
"""

import sklearn.preprocessing


preprocessors = [
    sklearn.preprocessing.OneHotEncoder(
        drop='if_binary', handle_unknown='ignore'),
    sklearn.preprocessing.OneHotEncoder(
        min_frequency=0.05, handle_unknown='infrequent_if_exist')
]
""" List of categorical transformers """
