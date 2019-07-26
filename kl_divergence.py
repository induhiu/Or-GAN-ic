""" Implementation of the Kullback-Leibler Divergence Score that we used to
    score our results. Adapted from stackexchange. """

# CITE: https://datascience.stackexchange.com/questions/9262/calculating-kl-divergence-in-python
# DETAILS: Implementation of Kullback-Leibler Divergence Score

import numpy as np
def KL(a, b):
    a = np.array(a, dtype=np.float)
    b = np.array(b, dtype=np.float)
    return np.sum(np.where(a != 0, a * np.log(a / b), 0))
