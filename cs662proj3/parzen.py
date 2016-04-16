#parzen.py
#function of parzen window
import math
import numpy as np

def window_function(sample,test_data,h):
    #print sample
    #print "x",test_data

    for x,x_i in zip(sample,test_data):

        if abs(x-x_i) > float(h)/2:
            return 0
    return 1

def estimate_single_px(samples,test_data,h):
    inRegion = 0
    outRegion = 0

    for sample in samples:
        if window_function(sample,test_data,h) == 1:
            inRegion  += 1
        else:
            outRegion += 1

    return float(inRegion)/((inRegion+outRegion)*(h**len(test_data)))


def get_px(samples,test_datas,h):
    px =[]

    for x in test_datas:
        x =np.array(x)

        val = estimate_single_px(samples,x,h)
        px.append(val)
    return px
