import numpy as np
import matplotlib.pyplot as plt
import math

def gradient_descent(theta,X,y,J,grad_J, alpha=.0001,max_iter=500000,cost_cutoff=.0000001):
    iter_count=0
    cost_delta=cost_cutoff+1
    cost=J(theta,X,y)
    while iter_count<max_iter and cost_delta>cost_cutoff:
        iter_count+=1
        theta=theta-alpha*grad_J(theta,X,y)
        new_cost=J(theta,X,y)
        cost_delta=abs(cost-new_cost)
        cost=new_cost
    return theta

def sigmoid(X):
    return (1 / (1 + np.exp(-X))) # YOUR CODE HERE

def pred(theta, X):
    return (sigmoid(np.dot(theta, X.T))) # YOUR CODE HERE

# J(𝜃)=□(64&1/2𝑚)∑_(𝑖=1)^𝑚▒(𝒚 ̂(𝑖)−𝒚(𝑖))2+𝜆𝜽2

def cost(theta,X,y):
    m=np.size(y)
    # return np.sum(np.dot((pred(theta, X)- y),(pred(theta, X)- y))) /2 /m
    return -np.sum(np.dot(y,np.log(pred(theta, X))) + np.dot((1.0000001-y), np.log(1.0000001 - pred(theta, X))))/m #L1: + 0.01*np.sum(pred(theta,X))#L2:+ 0.001 * np.sum( np.dot(y, y))/2# YOUR CODE HERE


def grad_cost(theta,X,y):
    m=np.size(y)
    # return np.clip(np.dot((pred(theta,X)-y),X)/m, 1, 200)
    return (np.sum(np.dot((pred(theta, X) - y),X))/ m )


def add_bias(X):
    retval=np.ones((X.shape[0],X.shape[1]+1))
    for i in range(1,X.shape[1]+1):
        retval[:,i]=X[:,i-1]
    return retval

# read input data
csv=np.genfromtxt('data.csv', delimiter=',')
X=csv[1:,0:2]
y=csv[1:,2]

ones=np.zeros(((int(np.sum(y)),2)))
zeros=np.zeros((int(y.shape[0]-np.sum(y)),2))

next1=0
next0=0
for i in range(y.shape[0]):
    if y[i]==0:
        zeros[next0,:]=X[i,:]
        next0+=1
    else:
        ones[next1,:]=X[i,:]
        next1+=1

X=add_bias(X)
theta_init=np.array([0,0,0])
print("Initial cost:", cost(theta_init,X,y))
print("Initial gradient:", grad_cost(theta_init,X,y))
theta_final=gradient_descent(theta_init,X,y,cost,grad_cost, alpha=.000000003)

print("Final parameters:", theta_final)

# plot points for reference
fig, ax = plt.subplots()
ax.scatter(zeros[:,0],zeros[:,1],color=[1,0,0])
ax.scatter(ones[:,0],ones[:,1],color=[0,0,1])
xs=np.array([0,(.5-theta_final[0])/theta_final[1]])
ys=np.array([(.5-theta_final[0])/theta_final[2],0])
print(xs)
print(ys)
ax.plot(xs,ys, color=[0,1,0])
ax.set(xlabel="x1", ylabel="x2", title="Classification in 2 Variables")
ax.grid()
plt.show()