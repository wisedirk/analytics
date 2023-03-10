import numpy as np

def act(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    

X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# err = (y - l2)^2
# l2 = act(sum(l1 * syn1))
# l1 = act(sum(l0 * syn0)
# derr_dsyn0[0] = derr_dl2 * dl2_dl1 * dl1_dsyn0[0]
# derr_dl2 = 2 * (y - l2) * -1 = - 2 * l2_error
# dl2_dl1 = dact_dsum * dsum_dl1
# dact_dsum = act(l2, deriv=True)
# dsum_dl1 = syn1
# dl1_dsyn0[0] = dact_dsum * dsum_dsyn0[0]
# dact_dsum = act(l1, deriv=True)
# dsum_dsyn0[0] = l0[0]
# derr_dsyn0[0] = -2 * l2_error * act(l2, deriv=True) * syn1 * act(l1, deriv=True) * l0[0] 

for i in range(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = act(np.dot(l0,syn0))
    l2 = act(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (i % 10000) == 0:
        print(f"Error: {np.mean(np.abs(l2_error))}")
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * act(l2, deriv=True)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = np.dot(l2_delta, syn1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * act(l1, deriv=True)

    syn1 += np.dot(l1.T, l2_delta)
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(f"{l1=}")
print(f"{l2=}")
print(f"{syn0=}")
print(f"{syn1=}")