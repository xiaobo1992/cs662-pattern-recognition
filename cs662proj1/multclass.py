import numpy as np
from scipy.stats import norm
from numpy import matrix
from numpy.linalg import linalg
import scipy.spatial.distance as distance
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import random


def compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,x):

    const = 0.5 * math.log(linalg.det(cov_2)/linalg.det(cov_1)) + math.log(float(w1)/w2)
    #print "const",const
    #print "xx",(0.5* (x-mu_2) * cov_2.I * (x-mu_2).T ) - (0.5* (x-mu_1) * cov_1.I * (x-mu_1).T )
    gx = (0.5 * (x-mu_2) * cov_2.I * (x-mu_2).T ) - (0.5* (x-mu_1) * cov_1.I * (x-mu_1).T )
    #print gx
    v = gx.tolist()
    return v[0][0]+const

    '''
    print v[0][0]+const
    const = math.log(float(w1)/w2)
    gx = (mu_1-mu_2)*cov_1.I*x.T+(mu_2 * cov_1.I * mu_2.T - mu_1 * cov_1.I * mu_1.T) /2
    print "gx",gx
    v  = gx.tolist()
    print v[0][0]+const
    '''

def compute_gix(mu,cov,wi,x):
    dim = len(x)
    x = matrix(x)
    mu = matrix(mu)
    cov = matrix(cov)
    part1 = math.log(wi)
    part2 = math.log(float(1)/math.pow(math.pi*2,dim/2)* linalg.det(cov))
    part3 = -0.5*(x-mu)*cov.I*(x-mu).T
    #print part1
    #print part2
    #print part3.tolist()[0][0]
    #print ""
    return part1+part2+part3.tolist()[0][0]

def get_label(mu,cov,wi,point):

    label = 0;
    val = 0;
    for i in range(len(mu)):
        #print i
        v = compute_gix(mu[i],cov[i],wi[i],point)
        #print i
        #print v[0][0]
        #print "   "
        if v > val:
            label = i
    return label

def classify(sample_points, mu,cov,wi):
    i = 0
    errors = []
    for samples in sample_points:
        #print "sample",i
        error = 0
        #l = []
        for point in samples:
            #print point
            label = get_label(mu,cov,wi,point)
            #l.append(label)
            if label != i:
                error+= 1
        errors.append(error)
        #print l
        i += 1;
    print errors

def create_mean(classes, dimension,minv,maxv):
    mu = []
    for i in range(classes):
            mu_i = []
            for j in range(dimension):
                mu_i.append(random.randint(minv,maxv))
            mu.append(mu_i)
    return mu

def create_covariance(classes, dimension,minv,maxv):
    covariances = []
    for i in range(classes):
            covariance = []
            for j in range(dimension):
                x = []
                for k in range(dimension):
                    if k == j:
                        x.append(random.randint(minv,maxv))
                    else:x.append(0)
                covariance.append(x)
            covariances.append(covariance)
    return covariances

def create_sample_point(classes, size, wi, mu, cov):
    sample_points = []

    for i in range(classes):
            points = np.random.multivariate_normal(mu[i], matrix(cov[i]), int(size*wi[i]))
            sample_points.append(points)
    return sample_points


if __name__ == "__main__":
    wi = [0.3,0.6,0.1,0.3,0.1,0.2]
    #mu = create_mean(3,2,1,10)
    mu = create_mean(3,2,1,500)
    print mu
    #cov = create_covariance(3,2,1,10)
    cov = create_covariance(3,2,1,500)
    print cov
    samples = create_sample_point(3, 10000, wi, mu,cov)

    color = ['red','green','blue','yellow','orange','pink']
    i = 0
    #print samples
    for sample in samples:
        x,y = sample.T
        plt.scatter(x,y,alpha = 0.6, color = color[i])
        i+= 1

    classify(samples, mu,cov,wi)
    plt.show()
