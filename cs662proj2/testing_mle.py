import mle
import numpy as np
import random
import parzen
from numpy import matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
import math
from numpy.linalg import linalg
import time


def compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,x):

    mu_1 =matrix(mu_1)
    mu_2 = matrix(mu_2)
    cov_1 = matrix(cov_1)
    cov_2 = matrix(cov_2)


    const = 0.5 * math.log(linalg.det(cov_2)/linalg.det(cov_1)) + math.log(float(w1)/w2)

    #print "const",const
    #print "xx",(0.5* (x-mu_2) * cov_2.I * (x-mu_2).T ) - (0.5* (x-mu_1) * cov_1.I * (x-mu_1).T )
    gx = (0.5 * (x-mu_2) * cov_2.I * (x-mu_2).T ) - (0.5* (x-mu_1) * cov_1.I * (x-mu_1).T )
    #print gx
    v = gx.tolist()
    return v[0][0]+const



def testing(train_samples_1,test_samples_1,train_samples_2,test_samples_2):

    now = time.time()


    estimate_mu_1 = mle.estimate_mu(train_samples_1)
    estimate_cov_1 = mle.estimate_covariance(train_samples_1)

    estimate_mu_2 = mle.estimate_mu(train_samples_2)
    estimate_cov_2 = mle.estimate_covariance(train_samples_2)

    w1 = len(train_samples_1)/float(len(train_samples_1)+len(train_samples_2))
    w2 = len(train_samples_2)/float(len(train_samples_1)+len(train_samples_2))

    error = 0
    for d in test_samples_1:
        val = compute_gx(estimate_mu_1 ,estimate_mu_2,estimate_cov_1,estimate_cov_2,w1,w2,d)
        if val > 0:
            pass
        else:
            error += 1

    for d in test_samples_2:
        val = compute_gx(estimate_mu_1 ,estimate_mu_2,estimate_cov_1,estimate_cov_2,w1,w2,d)
        if val > 0:
            error += 1
        else:
            pass

    res =  float(error)/float(len(test_samples_1)+len(test_samples_2))

    return res
    #errors.append(res)
    #times.append(time.time()-now)

    #print errors
    #print "error rate:",sum(errors)/len(errors)
    #print "average_time:",sum(times)/len(times)
    #return sum(errors)/len(errors)
    #return sum(times)/len(times)
