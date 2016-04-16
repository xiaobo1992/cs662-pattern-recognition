
'''



'''


def create_variance_matrix(d):
    cov = []
    for i in range(d):
        vec = []
        for j in range(d):
            if i == j:
                #vec.append(random.randint(1, 3))
                vec.append(1)
            else:
                vec.append(0)
        cov.append(vec)
    return matrix(cov)

def part2_A(d, equal):
    mu_1 = [0]* d
    mu_2 = [0]* d

    w1 = 0.5
    w2 = 0.5
    #size = 10000

    
    mean_2 = matrix(mu_2)
    cov_1 = create_variance_matrix(d)
    #cov_2 = create_variance_matrix(d)
    cov_2 = cov_1
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


                w1 = 0.5
                temp = []
                for i in range(size):
                    if random.random() < w1:
                        temp.append(0)
                    else:
                        temp.append(1)

                size1 = temp.count(0)
                size2 = temp.count(1)



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
            avg = sum(errors)/len(errors)
            print avg
            print float(avg)/size
            avg_error.append(float(avg)/size)
        print avg_error
        plt.plot([1,2,4,8,16],avg_error)
        plt.xlabel("distance")
        plt.ylabel("average error rate")
        plt.title("dimension 27 average error vs distance")
        plt.show()






if __name__ == "__main__":
    #part1_A()
    #part1_B() #doing 2d with same covariance and different covariance
    #part2_A(27) #doing 27d
    part2_B(27)
