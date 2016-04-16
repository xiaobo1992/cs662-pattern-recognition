#mle py
#function of mle

import numpy as np

def estimate_mu(data):
    data =  np.array(data)
    return np.mean(data,axis=0)

def estimate_covariance(data):
    data =  np.array(data)
    return np.cov(data.T)
