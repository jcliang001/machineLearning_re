import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(theta,X,y,J,grad_J,alpha=.001,max_iter=50000,cost_cutoff=.000001):
    iter_count=0
    cost_delta=cost_cutoff+1
    cost=J(theta,X,y)
    while iter_count<max_iter and cost_delta>cost_cutoff:
        iter_count+=1
        #print("Gradient:", grad_J(theta,X,y))
        theta=theta-alpha*grad_J(theta,X,y)
        new_cost=J(theta,X,y)
        cost_delta=abs(cost-new_cost)
        #print("\t\tdCost:", cost_delta)
        cost=new_cost
    print("\tCost:", cost)
    print("\tIterations:", iter_count)
    return theta

def pred(theta, X):
    return np.dot(theta,X)

def cost(theta,X,y):
    m=np.size(y)
    return .5*np.sum(np.square(pred(theta,X)-y))/m

def grad_cost(theta,X,y,clip=5):
    m=np.size(y)
    return np.clip(np.dot((pred(theta,X)-y),X.T)/m,-clip,clip)

def prep_poly_features(theta, x):
    retval=np.zeros((theta.shape[0],x.shape[0]))
    for i in range(theta.shape[0]):
        retval[i]=x**i
    return retval

np.random.seed(0) # Set the seed for consistent results between runs

x=np.arange(-4,12,.5) # Set the x variables
y=x*x-5*x+3 # This is the underlying function for the data - a polynomial of degree 2
for i in range(y.size): # Add some random noise to the data
    y[i]+=10*np.random.normal()

dimensions=2
theta=np.zeros((dimensions+1))
for i in range(dimensions+1):
    theta[i]=np.random.normal()

# Set up plotting...
fig, ax = plt.subplots()
ax.scatter(x,y,color=[1.,.5,0.])
x_init=np.arange(-4.,12.,.01)
ax.set(xlabel="x", ylabel="y", title="Polynomial Regression in 1 Variable")
ax.grid()

# Set up polynomial features
X=prep_poly_features(theta,x)
theta_final = gradient_descent(theta,X,y,cost,grad_cost, max_iter=102000, alpha=0.001)
print("Initial Parameters:", theta)
print("Final Parameters:",theta_final)
X_init = prep_poly_features(theta_final, x_init)
y_final=pred(theta_final,X_init)
ax.plot(x_init,y_final, color=[0,0,1], label="P(x)")

x1=prep_poly_features(theta, x_init)
y1=pred(theta,x1)
ax.plot(x_init,y1,color=[1,0,0], label="Q(x)")

theta2=np.array([3,-5,1])
x2=prep_poly_features(theta2, x_init)
y2=pred(theta2,x2)
ax.plot(x_init,y2,color=[1,.5,0], label="R(x)")

ax.legend()
plt.show()