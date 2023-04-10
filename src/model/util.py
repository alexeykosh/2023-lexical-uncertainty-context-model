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

def generate_lexicons(n_words, n_meanings):
    arrays =  np.array([list(map(int, list(np.binary_repr(i, width=n_words*n_meanings)))) 
                        for i in range(2**(n_words*n_meanings))])
    lexicons = arrays.reshape((2**(n_words*n_meanings), n_words, n_meanings))
    return lexicons[lexicons.sum(axis=1).min(axis=1) > 0]
