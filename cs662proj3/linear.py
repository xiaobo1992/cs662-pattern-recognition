import numpy as np
import math


def separate_data(train_data,class_label):
    train_1 = []
    train_2 = []

    for data,label in zip(train_data,class_label):
        if label == 0:
            train_1.append(np.array(data).T)
        else:
            train_2.append(np.array(data).T)
    train_1 = np.matrix(train_1)
    train_2 = np.matrix(train_2)
    return train_1,train_2


def compute_var(data,mean):
    val = 0
    for d in data:
        #print d,mean
        a = d-mean

        #print a.T*a
        val += np.dot(a.T,a)
    return val

def train(train_data, class_label):
    train_1,train_2 = separate_data(train_data,class_label)
    mean_1 = np.mean(train_1,axis=0)
    mean_2 = np.mean(train_2,axis=0)
    mean = np.mean(train_data,axis=0)


    s1_w = compute_var(train_1,mean_1)
    #print s1_w
    s2_w = compute_var(train_2,mean_2)
    #print s2_w
    sw = s1_w + s2_w
    w = sw.I*np.matrix(mean_1-mean_2).I
    #print np.linalg.norm(w,0)
    return w,mean

def test(test_data,w,mean):
    res = []
    #print w
    #print mean
    for d in test_data:
        
        v = (np.matrix(d-mean)*w)[0,0]
        #print v
        if v > 0:
            res.append(0)
        else:
            res.append(1)
    return res
