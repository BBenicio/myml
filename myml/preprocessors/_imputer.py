"""
Data preprocessors for features with missing values.

The `imputers` list contains scikit-learn compatible transformers
that impute missing features.
"""

from sklearn.impute import SimpleImputer


imputers = [
    SimpleImputer(strategy='mean'),
    SimpleImputer(strategy='median'),
    SimpleImputer(strategy='most_frequent')
]
""" List of imputers to fill missing values """
