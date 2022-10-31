from sklearn.impute import SimpleImputer


imputers = [
    SimpleImputer(strategy='mean'),
    SimpleImputer(strategy='median'),
    SimpleImputer(strategy='most_frequent')
]