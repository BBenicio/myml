import sklearn.preprocessing


preprocessors = [
    sklearn.preprocessing.FunctionTransformer(),
    sklearn.preprocessing.MaxAbsScaler(),
    sklearn.preprocessing.MinMaxScaler(),
    sklearn.preprocessing.StandardScaler(),
    sklearn.preprocessing.PowerTransformer(),
    sklearn.preprocessing.RobustScaler()
]