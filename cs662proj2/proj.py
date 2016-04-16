import numpy as np
from scipy.stats import norm
from numpy import matrix
from numpy.linalg import linalg
import scipy.spatial.distance as distance
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import random

'''
this function is used to create covariance matrix for mulitple dimension
'''
def create_variance_matrix(d):
    cov = []
    for i in range(d):
        vec = []
        for j in range(d):
            if i == j:
                vec.append(random.randint(1, 10))
                #vec.append(1)
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

'''
this function is used to generate random points according the probablity
'''
def generate_random_point(p,size):
    temp = []
    for i in range(size):
        if random.random() < p:
            temp.append(0)
        else:
            temp.append(1)
    size1 = temp.count(0)
    size2 = temp.count(1)
    return size1,size2


'''
this function is used to compute descrimante function gx
'''
def compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,x):

    const = 0.5 * math.log(linalg.det(cov_2)/linalg.det(cov_1)) + math.log(float(w1)/w2)
    #print "const",const
    #print "xx",(0.5* (x-mu_2) * cov_2.I * (x-mu_2).T ) - (0.5* (x-mu_1) * cov_1.I * (x-mu_1).T )
    gx = (0.5 * (x-mu_2) * cov_2.I * (x-mu_2).T ) - (0.5* (x-mu_1) * cov_1.I * (x-mu_1).T )
    #print gx
    v = gx.tolist()
    return v[0][0]+const


    print v[0][0]+const
    const = math.log(float(w1)/w2)
    gx = (mu_1-mu_2)*cov_1.I*x.T+(mu_2 * cov_1.I * mu_2.T - mu_1 * cov_1.I * mu_1.T) /2
    #print "gx",gx
    v  = gx.tolist()
    print v[0][0]+const



'''
part1_B is doing given fixed distance find the error rate with sample size in 2D
with covariance equal and not equal
'''

def part1_B(equal):
    w1 = 0.5
    w2 = 0.5

    avg_error = []

    if equal:
        #cov_1 = matrix(([5,1.2],[1.2,3]))
        cov_1 = create_variance_matrix(2)
        cov_2 = cov_1
    else:
        #cov_1 = matrix(([5,1.2],[1.2,3]))
        #cov_2 = matrix(([2,1.5],[1.5,2]))
        cov_1 = create_variance_matrix(2)
        cov_2 = create_variance_matrix(2)

    print cov_1
    print cov_2

    for size in [5,10,15,20,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1200,1500,3000]:
        print "size: ",size

        for distance in [4]:
            print "distance: ",distance
            mu_1 = matrix([2,distance])
            mu_2 = matrix([2,0])


            errors = []

            for time in range(10):


                size1,size2 = generate_random_point(w1,size)

                #p_X,p_Y = np.random.multivariate_normal(mu_1.tolist()[0], cov_1, int(w1*size)).T
                #n_X,n_Y = np.random.multivariate_normal(mu_2.tolist()[0], cov_2, int(w2*size)).T
                p_X,p_Y = np.random.multivariate_normal(mu_1.tolist()[0], cov_1, size1).T
                n_X,n_Y = np.random.multivariate_normal(mu_2.tolist()[0], cov_2, size2).T
                #print "cov_1",cov_1
                #print "cov_2",cov_2

                #px = plt.scatter(p_X,p_Y ,color='red',alpha = 0.5,label= "class 1")
                #py = plt.scatter(n_X,n_Y,color='blue',alpha = 0.5,label= "class 2")
                #print "distance: ",distance, ", mu_1: ",mu_1.tolist(),"mu_2: ",mu_2.tolist(),"cov1: ",cov_1.tolist(),"cov2: ",cov_2.tolist()

                #title = "distance: ",str(distance), ", mu_1: ",str(mu_1.tolist()),", mu_2: ",str(mu_2.tolist()),",\n cov1: ",str(cov_1.tolist()),", cov2: ",str(cov_2.tolist())
                #title = ''.join(title)
                    #print title
                #plt.title(title)
                #plt.legend(['class 1','class 2'])
                #plt.show()

                #plt.show()

                pos = 0
                neg = 0
                eq = 0

                error = 0
                for i in range(len(p_X)):
                    #point = create_random_point(2)
                    #print "point",point

                    #pt = matrix(point)
                    pt = matrix((p_X[i],p_Y[i]))
                    #print pt
                    val =  compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,pt)

                    #print "val",val
                    if val > 0:
                        #pos += 1
                        pass
                    elif val < 0:
                        neg += 1
                        error += 1
                    else:
                        eq +=  1

                for i in range(len(n_X)):

                    #pt = matrix(point)
                    pt = matrix((n_X[i],n_Y[i]))
                    val =  compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,pt)
                    #print "val",val
                    if val > 0:
                        #pos += 1
                        error += 1
                        pos += 1
                    elif val < 0:
                        pass
                    else:
                        eq +=  1

                errors.append(error)

            print errors
            avg = sum(errors)/float(len(errors))
            print avg
            print float(avg)/size
            avg_error.append(float(avg)/size)
    print avg_error
    plt.plot([5,10,15,20,50,75,100,150,200,250,300,400,500,600,700,800,900,1000,1200,1500,3000],avg_error)
    plt.xlabel("size")
    plt.ylabel("average error rate")
    plt.ylim([0,max(avg_error)*2])
    if equal:
        plt.title("2d average error vs size cov1 == cov2")
    else:
        plt.title("2d average error vs size cov1 != cov2")
    plt.show()


'''
part1_A is doing given fixed sample size find
the error rate with different mean distance 2d

'''

def part1_A(equal):
    w1 = 0.5
    w2 = 0.5
    #size = 1000
    if equal:
        #cov_1 = matrix(([5,1.4],[1.4,3]))
        cov_1 = create_variance_matrix(2)
        cov_2 = cov_1
    else:
        #cov_1 = matrix(([5,1.4],[1.4,3]))
        #cov_2 = matrix(([2,1.1],[1.1,2]))
        cov_1 = create_variance_matrix(2)
        cov_2 = create_variance_matrix(2)

    print cov_1
    print cov_2

    for size in [1000]:
        print "size: ",size
        avg_error = []


        for distance in [1,2,4,8,16]:
            print "distance: ",distance
            mu_1 = matrix([2,distance])
            mu_2 = matrix([2,0])


            #cov_1 = matrix(([5,0],[0,3]))

            errors = []

            for time in range(10):

                size1,size2 = generate_random_point(w1,size)

                #p_X,p_Y = np.random.multivariate_normal(mu_1.tolist()[0], cov_1, int(w1*size)).T
                #n_X,n_Y = np.random.multivariate_normal(mu_2.tolist()[0], cov_2, int(w2*size)).T
                p_X,p_Y = np.random.multivariate_normal(mu_1.tolist()[0], cov_1, size1).T
                n_X,n_Y = np.random.multivariate_normal(mu_2.tolist()[0], cov_2, size2).T
                #print "cov_1",cov_1
                #print "cov_2",cov_2

                px = plt.scatter(p_X,p_Y,color='red',alpha = 0.5,label= "class 1")
                py = plt.scatter(n_X,n_Y,color='blue',alpha = 0.5,label= "class 2")

                title = "distance: ",str(distance), ", mu_1: ",str(mu_1.tolist()),", mu_2: ",str(mu_2.tolist()),",\n cov1: ",str(cov_1.tolist()),", cov2: ",str(cov_2.tolist())
                title = ''.join(title)

                plt.title(title)
                plt.legend(['class 1','class 2'])
                plt.show()


                pos = 0
                neg = 0
                eq = 0

                error = 0
                for i in range(len(p_X)):
                    #point = create_random_point(2)
                    #print "point",point

                    #pt = matrix(point)
                    pt = matrix((p_X[i],p_Y[i]))

                    val =  compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,pt)

                    #print "val",val
                    if val > 0:
                        #pos += 1
                        pass
                    elif val < 0:
                        neg += 1
                        error += 1
                    else:
                        eq +=  1

                for i in range(len(n_X)):
                    #point = create_random_point(2)
                    #print "point",point

                    #pt = matrix(point)
                    pt = matrix((n_X[i],n_Y[i]))

                    val =  compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,pt)
                    #print "val:",val
                    if val > 0:
                        #pos += 1
                        error += 1
                        pos += 1
                    elif val < 0:
                        pass
                    else:
                        eq +=  1

                errors.append(error)

            print errors
            avg = sum(errors)/float(len(errors))
            print avg
            print float(avg)/size
            avg_error.append(float(avg)/size)
        print avg_error
        plt.plot([1,2,4,8,16],avg_error)
        plt.xlabel("distance")
        plt.ylabel("average error rate")
        plt.ylim([0,max(avg_error)*2])
        if equal:
            plt.title("2d average error vs distance cov1 == cov2")
        else:
            plt.title("2d average error vs distance cov1 != cov2")
        plt.show()


'''
This function is used to do given fixed sample size find the error rate with mean distance in N-d
'''

def part2_A(d,equal):
    mu_1 = [0]* d
    mu_2 = [0]* d

    w1 = 0.5
    w2 = 0.5


    mean_2 = matrix(mu_2)

    if equal:
        cov_1 = create_variance_matrix(d)
        cov_2 = cov_1
    else:
        cov_1 = create_variance_matrix(d)
        cov_2 = create_variance_matrix(d)

    print cov_1
    print cov_2
    print "dimesion: ", d
    for size in [1000]:
        print "size: ",size
        avg_error = []
        for distance in [1,2,4,8,16]:
            print "distance: ",distance
            mu_1[1] = distance
            #print mu_1
            mean_1 = matrix(mu_1)

            errors = []
            for time in range(10):
                size1,size2 = generate_random_point(w1,size)

                #p_X = np.random.multivariate_normal(mean_1.tolist()[0], cov_1, int(w1*size))
                #n_X = np.random.multivariate_normal(mean_2.tolist()[0], cov_2, int(w2*size))

                p_X = np.random.multivariate_normal(mean_1.tolist()[0], cov_1, size1)
                n_X = np.random.multivariate_normal(mean_2.tolist()[0], cov_2, size2)

                pos = 0
                neg = 0
                eq = 0

                error = 0
                #classify possitive point
                for i in range(len(p_X)):
                    #point = create_random_point(2)
                    #print "point",point

                    #pt = matrix(point)
                    pt = matrix((p_X[i]))
                    #print pt
                    val =  compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,pt)

                    #print "val",val
                    if val > 0:
                        #pos += 1
                        pass
                    elif val < 0:
                        neg += 1
                        error += 1
                    else:
                        eq +=  1

                #classify negative pointe
                for i in range(len(n_X)):
                    #point = create_random_point(2)
                    #print "point",point

                    #pt = matrix(point)
                    pt = matrix((n_X[i]))
                    val =  compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,pt)
                    #print "val",val
                    #determine erroor
                    if val > 0:
                        #pos += 1
                        error += 1
                        pos += 1
                    elif val < 0:
                        pass
                    else:
                        eq +=  1

                errors.append(error)

            print errors
            avg = sum(errors)/float(len(errors))
            print avg
            print float(avg)/size
            avg_error.append(float(avg)/size)
        print avg_error
        plt.plot([1,2,4,8,16],avg_error)
        plt.xlabel("distance")
        plt.ylabel("average error rate")
        plt.ylim([0,max(avg_error)*2])
        if equal:
            tit = "dimension: ",str(d)," average error vs distance cov1 == cov2"
            tit = "".join(tit)
            plt.title(tit)
        else:
            tit = "dimension: ",str(d)," average error vs distance cov1 != cov2"
            tit = "".join(tit)
            plt.title(tit)
            #plt.title("dimension 27: average error vs distance cov1 != cov2")
        plt.show()

'''
This funcion is doing given fixed distance finding error rate with different sample size in N-D

'''
c_index = 0
def part2_B(d,equal):
    mu_1 = [0]* d
    mu_2 = [0]* d

    global c_index
    colors = ['r', 'b', 'g', 'k', 'm']

    w1 = 0.5
    w2 = 0.5
    #size = 10000

    mean_2 = matrix(mu_2)
    if equal:
        cov_1 = create_variance_matrix(d)
        cov_2 = cov_1
    else:
        cov_1 = create_variance_matrix(d)
        cov_2 = create_variance_matrix(d)

    print cov_1
    print cov_2
    print "dimesion: ", d
    avg_error = []
    for size in [5,10,15,20,50,75,100,150,200,250,300,400,500,700,900,1000,1500,2000,2500,3000,5000,10000]:
        print "size: ",size
        for distance in [4]:
            print "distance: ",distance
            mu_1[1] = distance
            #print mu_1
            mean_1 = matrix(mu_1)

            errors = []
            for time in range(10):

                size1,size2 = generate_random_point(w1,size)

                p_X = np.random.multivariate_normal(mean_1.tolist()[0], cov_1, size1)
                n_X = np.random.multivariate_normal(mean_2.tolist()[0], cov_2, size2)
                #p_X = np.random.multivariate_normal(mean_1.tolist()[0], cov_1, int(w1*size))
                #n_X = np.random.multivariate_normal(mean_2.tolist()[0], cov_2, int(w2*size))

                pos = 0
                neg = 0
                eq = 0

                error = 0
                #classify possitive point
                for i in range(len(p_X)):
                    #point = create_random_point(2)
                    #print "point",point

                    #pt = matrix(point)
                    pt = matrix((p_X[i]))
                    #print pt
                    val =  compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,pt)

                    #print "val",val
                    if val > 0:
                        #pos += 1
                        pass
                    elif val < 0:
                        neg += 1
                        error += 1
                    else:
                        eq +=  1

                #classify negative pointe
                for i in range(len(n_X)):

                    #pt = matrix(point)
                    pt = matrix((n_X[i]))
                    val =  compute_gx(mu_1,mu_2,cov_1,cov_2,w1,w2,pt)
                    #print "val",val

                    #determine erroor
                    if val > 0:
                        #pos += 1
                        error += 1
                        pos += 1
                    elif val < 0:
                        pass
                    else:
                        eq +=  1

                errors.append(error)

            print errors
            avg = sum(errors)/float(len(errors))
            print avg
            print float(avg)/size
            avg_error.append(float(avg)/size)
    print avg_error
    plt.plot([5,10,15,20,50,75,100,150,200,250,300,400,500,700,900,1000,1500,2000,2500,3000,5000,10000],avg_error,label=str(d),color = colors[c_index])
    #plt.xlabel("size")
    #plt.ylabel("average error rate")
    #plt.ylim([0,max(avg_error)*2])
    c_index += 1
    if equal:
        tit = "dimension: ",str(d)," average error vs size, cov1 == cov2"
        tit = "".join(tit)
        plt.title(tit)
    else:
        tit = "dimension: ",str(d)," average error vs size, cov1 != cov2"
        tit = "".join(tit)
        plt.title(tit)
    #plt.show()


if __name__ == "__main__":
    #part1_A(True)
    #part1_A(False)
    #part1_B(True) #doing 27d with same covariance and different covariance
    #part1_B(False)


    #part2_A(2,True) #doing 27d
    #part2_A(5,True) #doing 27d
    #part2_A(10,True) #doing 27d
    #part2_A(27,True) #doing 27d
    #part2_A(2,False) #doing 27d
    #part2_A(5,False) #doing 27d
    #part2_A(10,False) #doing 27d
    #part2_A(27,False) #doing 27d
    #part2_B(2,True)
    #part2_B(5,True)
    #part2_B(10,True)
    #part2_B(15,True)
    #part2_B(27,True)
    part2_B(2,False)
    part2_B(5,False)
    part2_B(10,False)
    part2_B(15,False)
    part2_B(27,False)
    plt.xlabel("size")
    plt.ylabel("average error rate")
    plt.title("not diagnal cov1 != cov2")
    plt.legend(loc='best')
    plt.show()
