
# https://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np

# sigmoid function
def act(x,deriv=False):
    if(deriv==True):
        return x*(1-x) # https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
    return 1/(1+np.exp(-x))

# lin activation function works much better in this case!
# def act(x,deriv=False):
#     if(deriv==True):
#         return 0.1 * 1 # learning rate of 0.1
#     return x

    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

# err = (y - act(sum(syn0 * l0)))^2
# derr_dsyn0[0] = derr_dact * dact_dsum * dsum_dsyn0[0]
# err = (y - l1)^2 => derr_dact = 2 * (y - l1) * -1 = -2 * l1_error ~ - l1_error
# act = 1/(1-e^-x) => dact_dsum = l1/(1-l1) = act(l1, True)
# sum = syn0[0]*l0[0] + syn0[1]*l0[1] + syn0[2]*l0[2] => dsum_dsyn0[0] = l0[0]
# derr_dsyn0 = - l1_error * act(l1,True) * l0
# syn0 = syn0 - derr_dsyn0 = syn0 - - l1_error * act(l1,True) * l0 = syn0 + l1_error * act(l1,True) * l0     
# X[4,1] l0[4,3] syn0[3,1] l1[4,1] y[4,1] l1_error[4,1] l1_delta[4,1] l0.T*l1_delta[3,1]

for iter in range(10000):

    # forward propagation
    l0 = X
    l1 = act(np.dot(l0, syn0))

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * act(l1, True)

    # update weights
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(f"{l1=}")
print(f"{syn0=}")

