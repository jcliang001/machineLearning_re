import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


def shuffle(data,labels):
    n=data.shape[0]
    indices=np.random.permutation(n)
    return data[indices], labels[indices]

def split_train_test(data, labels, percent=0.8):
    n = data.shape[0]
    i=int(percent*n)
    return data[0:i], labels[0:i], data[i:n], labels[i:n]

def gradient_descent(theta1,theta2,X,y,J,grad_J,alpha=.001,max_iter=500000,cost_cutoff=.0000001):
    iter_count=0
    cost_delta=cost_cutoff+1
    cost=J(theta1,theta2,X,y)
    while iter_count<max_iter and cost_delta>cost_cutoff:
        iter_count+=1
        theta1_grad, theta2_grad=grad_J(theta1,theta2,X,y)
        theta1=theta1-alpha*theta1_grad
        theta2=theta2-alpha*theta2_grad
        new_cost=J(theta1,theta2,X,y)
        cost_delta=abs(cost-new_cost)
        cost=new_cost
    return theta1,theta2

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_grad(X):
    return None #YOUR CODE HERE

def add_bias(X):
    retval=np.ones((X.shape[0],X.shape[1]+1))
    for i in range(1,X.shape[1]+1):
        retval[:,i]=X[:,i-1]
    return retval

def pred(theta1, theta2, X):
    a2 = None  # YOUR CODE HERE, don't forget the hidden layer bias!
    a3 = None  # YOUR CODE HERE, no bias needed here
    return a3

def cost(theta1,theta2,X,y):
    m=np.size(y)
    Y=np.zeros((m,10))
    for i in range(m):
        Y[i,int(y[i])]=1
    retval=0
    y_hat = None # YOUR CODE HERE
    for i in range(10):
        retval-=None # YOUR CODE HERE - use cost function for logistic regression for the given output column
    return retval/m # Note the division by m - no need to do it on the previous line

def grad_cost(theta1, theta2, X,y):
    m=np.size(y)
    Y = np.zeros((m, 10))
    for i in range(m):
        Y[i, int(y[i])] = 1
    a2=None # YOUR CODE HERE (hint: you've done this before)
    a3=None # YOUR CODE HERE (hint: ditto)
    d3=None # YOUR CODE HERE
    d2=None # YOUR CODE HERE
    theta1_grad=None # YOUR CODE HERE
    theta2_grad=None # YOUR CODE HERE
    return theta1_grad, theta2_grad

mnist = loadmat("mnist-original.mat")
mnist_data = mnist["data"].T
mnist_labels = mnist["label"][0]
sh_data, sh_labels = shuffle(mnist_data, mnist_labels)

train_X, train_y, test_X, test_y = split_train_test(sh_data, sh_labels)

n=train_X.shape[0]-1
imgplot = plt.imshow(train_X[n].reshape((28,28)))
print("Label:", train_y[n])
plt.show()

train_X=add_bias(train_X/255.0)
test_X=add_bias(test_X/255.0)

hidden_size=32
classes=10

theta1=np.random.uniform(-0.12, .12, size=(np.size(train_X[1]),hidden_size))
theta2=np.random.uniform(-0.12, .12, size=(hidden_size+1,classes))

print("Random parameters:")
print ("\tLabels:", test_y[0:100])
preds=np.argmax(pred(theta1, theta2, test_X[0:100]), axis=1)
print ("\tPreds:", preds)
accuracy=np.sum(test_y[0:100]==preds)/100.
print ("\tAccuracy:", accuracy)


theta1, theta2 = gradient_descent(theta1, theta2, train_X, train_y, cost, grad_cost, alpha=.00003, max_iter=10)
print("\nAfter 10 iterations:")
print ("\tLabels:", test_y[0:100])
preds=np.argmax(pred(theta1, theta2, test_X[0:100]), axis=1)
print ("\tPreds:", preds)
accuracy=np.sum(test_y[0:100]==preds)/100.
print ("\tAccuracy:", accuracy)
print()


theta1, theta2 = gradient_descent(theta1, theta2, train_X, train_y, cost, grad_cost, alpha=.00001, max_iter=10)
preds=np.argmax(pred(theta1, theta2, test_X[0:100]), axis=1)
accuracy = np.sum(test_y[0:100] == preds) / 100.
print("\nAfter 20 iterations:")
print("\tAccuracy:", accuracy)
print()

theta1, theta2 = gradient_descent(theta1, theta2, train_X, train_y, cost, grad_cost, alpha=.000003, max_iter=10)
preds=np.argmax(pred(theta1, theta2, test_X[0:100]), axis=1)
accuracy = np.sum(test_y[0:100] == preds) / 100.
print("\nAfter 30 iterations:")
print("\tAccuracy:", accuracy)
print()

theta1, theta2 = gradient_descent(theta1, theta2, train_X, train_y, cost, grad_cost, alpha=.000001, max_iter=10)
preds = np.argmax(pred(theta1, theta2, test_X[0:100]), axis=1)
accuracy = np.sum(test_y[0:100] == preds) / 100.
print("\nAfter 40 iterations:")
print("\tAccuracy:", accuracy)
print()

theta1, theta2 = gradient_descent(theta1, theta2, train_X, train_y, cost, grad_cost, alpha=.0000003, max_iter=10)
print("\nAfter 50 iterations:")
print ("\tLabels:", test_y[0:100])
preds=np.argmax(pred(theta1, theta2, test_X[0:100]), axis=1)
print ("\tPreds:", preds)
accuracy=np.sum(test_y[0:100]==preds)/100.
print ("\tAccuracy:", accuracy)

preds=np.argmax(pred(theta1, theta2, train_X), axis=1)
accuracy=np.sum(train_y==preds)/np.size(train_y)
print("Training accuracy: ", accuracy)