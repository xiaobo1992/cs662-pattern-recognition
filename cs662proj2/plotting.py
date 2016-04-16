import matplotlib.pyplot as plt

def plot(title, parzen, mle,xtitle,ytitle,x_attr,h):
    plt.clf()
    plt.title(title)
    plt.xlabel(xtitle)
    plt.plot(x_attr,mle,label="MLE",color = 'r')
    plt.plot(x_attr,parzen,label="Parzen Window with "+str(h),color = 'b')
    plt.legend(loc='best')
    #plt.ylim(0,0.8)
    plt.show()
