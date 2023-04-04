import numpy as np

def safelog(vals):
    with np.errstate(divide='ignore'):
        return np.log(vals)

def normalize(A):
    # if only one dimension do not transpose
    if len(A.shape) == 1:
        return A / A.sum()
    with np.errstate(divide='ignore'):
        return np.nan_to_num((A.T / A.sum(axis=1)).T)

def lex_to_str(l):
    return ''.join(map(str, l.reshape(l.shape[0] * l.shape[1])))