from enum import IntEnum

class DataType(IntEnum):
    numeric = 1
    categorical = 2
    # ordinal = 3
    # textual = 4

class ProblemType(IntEnum):
    classification = 1
    regression = 2
