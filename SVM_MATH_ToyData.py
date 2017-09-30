import numpy as np
from matplotlib import pyplot as plt

#define data  -> (x, y, bias term)
X = np.array([
                [-2,4,-1],
                [4,1,-1],
                [1,6,-1],
                [2,4,-1],
                [6,2,-1]
            ])
#output labels
y = np.array([-1,-1,1,1,1])

#ploting data
# first 2 samples -> '-' marker
# next 3 sqamples -> '+' marker
plt.figure(0)
for d,sample in enumerate(X):
    if(d<2):
        plt.scatter(sample[0],sample[1], s=120, marker='_', linewidths=2)
    else:
        plt.scatter(sample[0],sample[1], s=120, marker='+', linewidths=2)
#draw randow decision boundary
plt.plot([-2,6],[6,0.5])
plt.show()
#perform stochaistic gradient descent to learn separating
# hyperplane

def svm_sgd(X,Y):
    #intitialize wt vectors with zero
    w = np.zeros(len(X[0]))

    #learnign rate
    eta = 1

    #no of iterations
    epochs = 100000

    #store misclassifications to see the change over time
    errors = []

    #let's train
    for epoch in range(1,epochs):
        error = 0
        if(epoch % 100 == 0 ):
            print(epoch)
        for i,x in enumerate(X):
            if(Y[i]*np.dot(X[i],w))<1:
                #misclassified
                #update weights
                w = w + eta * ((X[i]*Y[i]) + (-2* (1/epoch)*w ) )
                error = 1
            else:
                #correct classification - update weights
                w = w + eta * (-2* (1/epoch)*w )
        errors.append(error)

    #plot rate of classification error
    plt.figure(1)
    plt.plot(errors,'')
    plt.ylim(0.5,1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    return w

#train the model
weights = svm_sgd(X,y)

#plot the model
plt.figure(2)
for d,sample in enumerate(X):
    if(d<2):
        plt.scatter(sample[0],sample[1], s=120, marker='_', linewidths=2)
    else:
        plt.scatter(sample[0],sample[1], s=120, marker='+', linewidths=2)

#add test asmples
plt.scatter(2,2,s=120,marker='_',linewidths=2, color='yellow' )
plt.scatter(4,3,s=120,marker='+',linewidths=2, color='blue' )

#print the learned hyperplane
x2 = [weights[0],weights[1],-weights[1],weights[0]]
x3 = [weights[0],weights[1],weights[1], -weights[0]]

x2x3 = np.array([x2,x3])

X,Y,U,V = zip(*x2x3)
ax = plt.gca()

ax.quiver(X,Y,U,V,scale=1, color='blue')
plt.show()


