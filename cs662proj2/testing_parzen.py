import numpy as np
import random
import parzen
import time
from numpy import matrix


def classify(x,train_samples_1,train_samples_2,w1,w2,h):
    px_1 = parzen.estimate_single_px(train_samples_1,x,h)
    px_2 = parzen.estimate_single_px(train_samples_2,x,h)
    if px_1*w1 > px_2*w2:
        return 1
    else:
        return 2


def testing(train_samples_1,test_samples_1,train_samples_2,test_samples_2,h):
    errors = []
    times = []

    now = time.time()

    w1 = len(train_samples_1)/float(len(train_samples_1)+len(train_samples_2))
    w2 = len(train_samples_2)/float(len(train_samples_1)+len(train_samples_2))
    error = 0
    for x in test_samples_1:
        label = classify(x,train_samples_1,train_samples_2,w1,w2,h)
        if label == 1:
            pass
        else:
            error +=1

    for x in test_samples_2:
        label = classify(x,train_samples_1,train_samples_2,w1,w2,h)
        if label == 2:
            pass
        else:
            error +=1


    res = error/float(len(test_samples_1)+len(test_samples_2))
    return res
    #errors.append(res)
    #times.append(time.time()-now)

    #print errors
    #print "error rate:",sum(errors)/len(errors)
    #print "average_time:",sum(times)/len(times)
    #return sum(errors)/len(errors)
    #return sum(times)/len(times)
