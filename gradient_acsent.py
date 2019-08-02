import numpy as np
import matplotlib.pyplot as plt

# gradient descent function
# theta is an initial set of parameters
# X is a set of input values.  NOTE: 1's need to be inserted into the first position for biases to work
# y is a set of output values
# J is the cost function and needs to take (theta, X, y) as inputs
# grad_J is the gradient of the cost function and takes the same arguments as J
# alpha is the step size
# max_iter is the maximum iteration count
# cost_cutoff is the cost delta at which the algorithm will stop
def gradient_descent(theta,X,y,J,grad_J,alpha=.05,max_iter=100,cost_cutoff=.000001):
    iter_count=1 # YOUR CODE HERE - set to an initial value
    cost_delta=0.0003 # YOUR CODE HERE - set to a safe initial value
    cost=10000 # YOUR CODE HERE - set to the correct initial value
    while iter_count<max_iter and cost_delta>cost_cutoff:
        iter_count+=1 # YOUR CODE HERE - increment
        theta=theta - alpha*grad_J(theta, X, y ) # YOUR CODE HERE - core of the algorithm goes here
        new_cost=J(theta, X, y) # YOUR CODE HERE - calculate new cost
        cost_delta=abs(new_cost - cost) # YOUR CODE HERE - calculate delta (HINT: use abs(...)
        cost=new_cost # YOUR CODE HERE - set appropriate cost
        #print("\tIteration:", iter_count, "\tCost:", cost) #uncomment for debugging if needed
    return theta


def pred(theta, X):
    return np.matmul(theta,X)

def J(theta,X,y):
    m=np.size(y)
    return .5*np.sum(np.square(pred(theta,X)-y))/m

def grad_J(theta,X,y):
    m=np.size(y)
    return np.matmul((pred(theta,X)-y),X.T)/m

def add_biases(X):
    return np.array([np.ones((X.size)),X])

x=np.array([1.,3.,2.,4.])
y=np.array([1.,2.,3.,4.])
X=add_biases(x)
theta=np.array([4.,-2.])

fig, ax = plt.subplots()
ax.scatter(x,y,color=[1.,.5,0.])
x_init=np.arange(0.,5.,.01)
X_init=add_biases(x_init)
y_init=pred(theta,X_init)

ax.set(xlabel="x", ylabel="y", title="Applying Gradient Descent to a Simple Regression")
ax.grid()

ax.plot(x_init,y_init, color=[1.,0.,0.], label="Initial")

theta_temp = gradient_descent(theta, X, y, J, grad_J, max_iter=1)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.9,.9,1.], linestyle='--')

theta_temp = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=1)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.8,.8,1.], linestyle='--')

theta_temp = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=1)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.7,.7,1.], linestyle='--')

theta_temp = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=1)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.6,.6,1.], linestyle='--')

theta_temp = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=5)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.5,.5,1.], linestyle='--')

theta_temp = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=25)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.4,.4,1.], linestyle='--')

theta_temp = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=50)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.3,.3,1.], linestyle='--')

theta_temp = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=50)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.2,.2,1.], linestyle='--')

theta_temp = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=100)
y_temp=pred(theta_temp,X_init)
ax.plot(x_init,y_temp, color=[.1,.1,1.], linestyle='--')

theta_final = gradient_descent(theta_temp,X,y,J,grad_J, max_iter=10000)

y_final=pred(theta_final,X_init)
ax.plot(x_init,y_final, color=[0.,1.,0.], label="Final")

print("Final Parameters:", theta_final)
ax.legend()
plt.show()