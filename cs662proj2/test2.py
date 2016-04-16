import testing_mle
import testing_parzen
import plotting
from matplotlib.mlab import bivariate_normal
import matplotlib.pyplot as plt
import random
from numpy import matrix
import numpy as np

def create_variance_matrix(d):
    cov = []
    for i in range(d):
        vec = []
        for j in range(d):
            if i == j:
                vec.append(random.randint(0, 5))
                #vec.append(15)
            else:
                vec.append(0)
        cov.append(vec)

    for i in range(d):
        for j in range(i+1,d,1):
            if i != j:
                #cov[i][j] = random.randint(1, 3)
                cov[i][j] = round(random.random(),2)
                cov[j][i] = cov[i][j]
    return matrix(cov)

def generate_random_point(dimesion,distance):
    p1 = []
    p2 = []
    for t in range(dimesion):
        x = random.random()*10-10
        p1.append(x)
        p2.append(x)

    l = len(p1)
    i = int(random.random()*l)
    p2[i] += distance
    return p1,p2

def generate_normal_data(mean,cov,size):
    data = np.random.multivariate_normal(mean, cov, size)
    return data

def generate_possion_data(size,dimension):
    data = np.random.poisson(10,size=(size,dimension))
    return data

def generate_uniform_data(start,end,size,dimesion):
    data = np.random.uniform(start,end,size=(size,dimesion))
    return data

def sampling(data_1,data_2,prob):
    train_1 = []
    test_1 = []
    train_2 = []
    test_2 = []
    for d in data_1:
        if random.random() < prob:
            train_1.append(d)
        else:
            test_1.append(d)

    for d in data_2:
        if random.random() < prob:
            train_2.append(d)
        else:
            test_2.append(d)
    return train_1,test_1,train_2,test_2


#testing by trainning sampling size
dimensions = [2,5,10,15]
size = 1500
probs = [0.03,0.05,0.08,0.1,0.13,0.14,0.3,0.5,0.7]
for d in dimensions:

    #generate parameter
    mean_1,mean_2 = generate_random_point(d,3)
    cov_1 = create_variance_matrix(d)
    cov_2 = cov_1

    #generate data
    data_1 = generate_normal_data(mean_1,cov_1,size)
    #data_1 = generate_uniform_data(-10,10,size,d)
    data_2 = generate_normal_data(mean_2,cov_2,size)
    #data_2 = generate_uniform_data(-10,10,size,d)

    parzen_1 = []
    parzen_3 = []

    mle = []

    for prob in probs:
        print d,size*prob
        parzen_errors_1 = []
        parzen_errors_3 = []

        mle_errors = []
        for t in range(10):
            train_samples_1,test_samples_1,train_samples_2,test_samples_2 = sampling(data_1,data_2,prob)
            parzen_errors_1.append(testing_parzen.testing(train_samples_1,test_samples_1,train_samples_2,test_samples_2,1))
            parzen_errors_3.append(testing_parzen.testing(train_samples_1,test_samples_1,train_samples_2,test_samples_2,3))
            mle_errors.append(testing_mle.testing(train_samples_1,test_samples_1,train_samples_2,test_samples_2))
        parzen_1.append(sum(parzen_errors_1)/len(parzen_errors_1))
        parzen_3.append(sum(parzen_errors_3)/len(parzen_errors_3))

        mle.append(sum(mle_errors)/len(mle_errors))
    print parzen_1
    print parzen_3
    print mle

    title = "Parzen Window vs MLE in "+str(d)+"D"
    title = str(title);
    xtitle = "Size of trainning samples"
    ytitle = "error rate"
    x_attr = [x*size for x in probs]

    plotting.plot(title, parzen_1, mle,xtitle,ytitle,x_attr,"h = 1")
    plotting.plot(title, parzen_3, mle,xtitle,ytitle,x_attr,"h = 3")
