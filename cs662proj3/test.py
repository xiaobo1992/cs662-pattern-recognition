import svm
import linear
import numpy as np
import random
import matplotlib.pyplot as plt
import testing_mle
import testing_parzen

def generate_gaussian_data(mean,cov,size,label):
    x = np.random.multivariate_normal(mean, cov,size)
    y = np.repeat(label, size)
    return x,y

def error_rate(predict_label, true_label):
    error = 0
    accurate = 0
    for l1,l2 in zip(predict_label, true_label):
        if l1 == l2:
            accurate += 1
        else:
            error += 1
    return float(error)/float(accurate+error)
    #return error

def separate_data(prior,datas,labels):
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for data,label in zip(datas,labels):
        if random.random() < prior:
            train_data.append(data)
            train_label.append(label)
        else:
            test_data.append(data)
            test_label.append(label)
    return train_data,train_label,test_data,test_label

def create_covariance_matrix(dimension):
    cov = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(i,dimension,1):
            if i == j:
                cov[i][j] = random.random()*5
            else:
                val = 2*random.random()-1
                cov[i][j] = val
                cov[j][i] = val
    return cov

def set_up_parameter(dimension,distance):
    mean_1 = np.zeros(dimension)
    mean_2 = np.zeros(dimension)
    for d in range(dimension):
        val = random.random()
        mean_1[d] = val
        mean_2[d] = val

    index = int(random.random()*dimension)
    mean_2[index] += distance

    cov_1 = create_covariance_matrix(dimension)
    cov_2 = create_covariance_matrix(dimension)
    return mean_1,cov_1,mean_2,cov_2

def generate_unseparateable_data(size,dimension):
    mean_1,cov_1,mean_2,cov_2 = set_up_parameter(dimension,distance)



    x_1,y_1 = generate_gaussian_data(mean_1,cov_1,2000,0)
    x_2,y_2 = generate_gaussian_data(mean_1,cov_2,2000,1)

    #x_1 = 20*np.random.random((size,dimension))-10
    #x_2 = 20*np.random.random((size,dimension))-10
    #y_1 = np.repeat(0, size)
    #y_2 = np.repeat(1, size)
    return x_1,y_1,x_2,y_2

'''
testing
'''

dimension = 50
distance = 3
sizes = [10,20,50,100,200,300,400,500,600,800,1000,1200,1300,1700,2000]
dimensions = [25]
#dimensions = [2,3]
prior_1 = 0.5
prior_2 = 1 - prior_1


'''
mean_1,cov_1,mean_2,cov_2 = set_up_parameter(dimension,distance)
x_1,y_1 = generate_gaussian_data(mean_1,cov_1,10000,0)
x_2,y_2 = generate_gaussian_data(mean_2,cov_2,10000,1)
'''
'''
#testing training size vs error rate
for dimension in dimensions:

    print dimension,"dimension"
    mean_1,cov_1,mean_2,cov_2 = set_up_parameter(dimension,distance)
    x_1,y_1 = generate_gaussian_data(mean_1,cov_1,2000,0)
    x_2,y_2 = generate_gaussian_data(mean_2,cov_2,2000,1)
    svm_rbf_errors = []
    svm_linear_errors = []
    linear_errors = []
    svm_poly_errors = []
    mle_errors=[]
    parzen_errors =[]
    for size in sizes:

        print "size:",size

        #separate data
        s1 = size * prior_1
        s2 = size * prior_2

        train_data_1 = x_1[0:s1]
        train_label_1 = y_1[0:s1]
        train_data_2 = x_2[0:s2]
        train_label_2 = y_2[0:s2]

        test_data_1 = x_1[s1:]
        test_label_1 = y_1[s1:]
        test_data_2 = x_2[s2:]
        test_label_2 = y_2[s2:]


        for a in train_data_1:
            plt.scatter(a[0],a[1],color='blue')
        for a in train_data_2:
            plt.scatter(a[0],a[1],color='red')
        plt.show()


        #generate traing data
        x = np.concatenate((train_data_1,train_data_2), axis=0)
        y = np.concatenate((train_label_1,train_label_2), axis=0)


        #traning a model
        svm_rbf = svm.train(x,y,"rbf")
        svm_linear = svm.train(x,y,"linear")
        w,mean = linear.train(x,y)

        #generate test data
        test_data  = np.concatenate((test_data_1,test_data_2), axis=0)
        test_label = np.concatenate((test_label_1,test_label_2), axis=0)

        #prediction
        svm_rbf_label = svm.test(test_data,svm_rbf)
        linear_label = linear.test(test_data,w,mean)
        svm_linear_label = svm.test(test_data,svm_linear)

        #get result
        svm_rbf_error = error_rate(test_label,svm_rbf_label)
        linear_error = error_rate(test_label,linear_label)
        svm_linear_error = error_rate(test_label,svm_linear_label)
        mle_error = testing_mle.testing(train_data_1,test_data_1,train_data_2,test_data_2)
        parzen_error = testing_parzen.testing(train_data_1,test_data_1,train_data_2,test_data_2,3)
        print "svm error(rbf):",svm_rbf_error
        print "svm error(linear):",svm_linear_error
        print "linaer_classifer error:",linear_error
        print "mle error:",mle_error
        print "parzen error:",parzen_error
        #store results
        svm_rbf_errors.append(svm_rbf_error)
        linear_errors.append(linear_error)
        svm_linear_errors.append(svm_linear_error)
        mle_errors.append(mle_error)
        parzen_errors.append(parzen_error)
        print ""

    plt.title("Training size vs Error rate in "+str(dimension)+"D")
    plt.xlabel("Training size")
    plt.ylabel("Error rate")
    plt.plot(sizes,svm_rbf_errors,label="svm rbf")
    plt.plot(sizes,linear_errors,label="linear classifer")
    plt.plot(sizes,svm_linear_errors,label="svm linear")
    plt.plot(sizes,mle_errors,label="MLE")
    plt.plot(sizes,parzen_errors,label="Parzen window")

    plt.legend(loc="best")
    plt.show()
'''


#testing mean distance vs error rate
distances = [0,1,2,3,4,5,6,7,8,9,10]

for dimension in dimensions:
    print dimension,"dimension"

    svm_rbf_errors = []
    svm_linear_errors = []
    linear_errors = []
    svm_poly_errors = []
    mle_errors=[]
    parzen_errors =[]
    size = 1000
    for distance in distances:
        print distance,"distance"
        mean_1,cov_1,mean_2,cov_2 = set_up_parameter(dimension,distance)
        x_1,y_1 = generate_gaussian_data(mean_1,cov_1,1000,0)
        x_2,y_2 = generate_gaussian_data(mean_2,cov_2,1000,1)

        #separate data
        s1 = size * prior_1
        s2 = size * prior_2

        train_data_1 = x_1[0:s1]
        train_label_1 = y_1[0:s1]
        train_data_2 = x_2[0:s2]
        train_label_2 = y_2[0:s2]

        test_data_1 = x_1[s1:]
        test_label_1 = y_1[s1:]
        test_data_2 = x_2[s2:]
        test_label_2 = y_2[s2:]

        #generate traing data
        x = np.concatenate((train_data_1,train_data_2), axis=0)
        y = np.concatenate((train_label_1,train_label_2), axis=0)


        #traning a model
        svm_rbf = svm.train(x,y,"rbf")
        svm_linear = svm.train(x,y,"linear")
        w,mean = linear.train(x,y)

        #generate test data
        test_data  = np.concatenate((test_data_1,test_data_2), axis=0)
        test_label = np.concatenate((test_label_1,test_label_2), axis=0)

        #prediction
        svm_rbf_label = svm.test(test_data,svm_rbf)
        linear_label = linear.test(test_data,w,mean)
        svm_linear_label = svm.test(test_data,svm_linear)

        #get result
        svm_rbf_error = error_rate(test_label,svm_rbf_label)
        linear_error = error_rate(test_label,linear_label)
        svm_linear_error = error_rate(test_label,svm_linear_label)
        mle_error = testing_mle.testing(train_data_1,test_data_1,train_data_2,test_data_2)
        parzen_error = testing_parzen.testing(train_data_1,test_data_1,train_data_2,test_data_2,3)

        print "svm error(rbf):",svm_rbf_error
        print "svm error(linear):",svm_linear_error
        print "linaer_classifer error:",linear_error
        print "mle error:",mle_error
        print "parzen error:",parzen_error
        #store results
        svm_rbf_errors.append(svm_rbf_error)
        linear_errors.append(linear_error)
        svm_linear_errors.append(svm_linear_error)
        mle_errors.append(mle_error)
        parzen_errors.append(parzen_error)
        print ""

    plt.title("mean distance vs Error rate in "+str(dimension)+"D")
    plt.xlabel("mean distance")
    plt.ylabel("Error rate")
    plt.ylim(-0.1,0.6)
    plt.plot(distances,svm_rbf_errors,label="svm rbf")
    plt.plot(distances,linear_errors,label="linear classifer")
    plt.plot(distances,svm_linear_errors,label="svm linear")
    plt.plot(distances,mle_errors,label="MLE")
    plt.plot(distances,parzen_errors,label="Parzen window")
    plt.legend(loc="best")
    plt.show()



size = 1000
svm_rbf_errors = []
svm_linear_errors = []
linear_errors = []
svm_poly_errors = []
mle_errors=[]
parzen_errors =[]

'''
mean_1,cov_1,mean_2,cov_2 = set_up_parameter(25,0)

for dimension in dimensions:
    print dimension,"dimension"


    #size = 1000
    mean_1,cov_1,mean_2,cov_2 = set_up_parameter(dimension,6)
    sigma_1 = cov_1[0:dimension,0:dimension]
    sigma_2 = cov_2[0:dimension,0:dimension]

    x_1,y_1 = generate_gaussian_data(mean_1,sigma_1,1000,0)
    x_2,y_2 = generate_gaussian_data(mean_2,sigma_2,1000,1)


    #separate data
    s1 = size * prior_1
    s2 = size * prior_2

    train_data_1 = x_1[0:s1]
    train_label_1 = y_1[0:s1]
    train_data_2 = x_2[0:s2]
    train_label_2 = y_2[0:s2]

    test_data_1 = x_1[s1:]
    test_label_1 = y_1[s1:]
    test_data_2 = x_2[s2:]
    test_label_2 = y_2[s2:]

    #generate traing data
    x = np.concatenate((train_data_1,train_data_2), axis=0)
    y = np.concatenate((train_label_1,train_label_2), axis=0)


    #traning a model
    svm_rbf = svm.train(x,y,"rbf")
    svm_linear = svm.train(x,y,"linear")
    w,mean = linear.train(x,y)

    #generate test data
    test_data  = np.concatenate((test_data_1,test_data_2), axis=0)
    test_label = np.concatenate((test_label_1,test_label_2), axis=0)

    #prediction
    svm_rbf_label = svm.test(test_data,svm_rbf)
    linear_label = linear.test(test_data,w,mean)
    svm_linear_label = svm.test(test_data,svm_linear)

    #get result
    svm_rbf_error = error_rate(test_label,svm_rbf_label)
    print "svm error(rbf):",svm_rbf_error

    linear_error = error_rate(test_label,linear_label)
    print "linaer_classifer error:",linear_error


    svm_linear_error = error_rate(test_label,svm_linear_label)
    print "svm error(linear):",svm_linear_error

    mle_error = testing_mle.testing(train_data_1,test_data_1,train_data_2,test_data_2)
    print "mle error:",mle_error

    parzen_error = testing_parzen.testing(train_data_1,test_data_1,train_data_2,test_data_2,3)
    print "parzen error:",parzen_error

    #store results
    svm_rbf_errors.append(svm_rbf_error)
    linear_errors.append(linear_error)
    svm_linear_errors.append(svm_linear_error)
    mle_errors.append(mle_error)
    parzen_errors.append(parzen_error)
    print ""

plt.title("Dimension vs Error rate linear separable data with variance < 10000")
plt.xlabel("Dimension")
plt.ylabel("Error rate")
plt.ylim(-0.1,0.6)
plt.plot(dimensions,svm_rbf_errors,label="svm rbf")
plt.plot(dimensions,linear_errors,label="linear classifer")
plt.plot(dimensions,svm_linear_errors,label="svm linear")
plt.plot(dimensions,mle_errors,label="MLE")
plt.plot(dimensions,parzen_errors,label="Parzen window")
plt.legend(loc="best")
plt.show()
'''

'''
x_1,y_1,x_2,y_2 = generate_unseparateable_data(size,dimension)
for a in x_1:
    plt.scatter(a[0],a[1],color='blue')
for a in x_2:
    plt.scatter(a[0],a[1],color='red')
plt.show()
'''
