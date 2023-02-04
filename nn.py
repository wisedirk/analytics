import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def prop(x,weight,bias):
    return sigmoid(weight @ x + bias)



x = np.array([[1],
              [1],
              [0],
              [0],
              [1]])
print(x)

w1 = np.array([[0.1,0.2,0.3,0.4,0.5],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0]])
print(w1)

b1 = np.array([[0.1],
                 [0.2],
                 [0.3],
                 [0.4],
                 [0.5],
                 [0.6],
                 [0.7]])
print(b1)

s1 = np.dot(w1, x) + b1
print(s1)
s1 = w1 @ x + b1
print(s1)

y1 = sigmoid(s1)
print(y1)

w2 = np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0],
                   [0,0,0,0,0,0,0]])

b2 = np.array([[0],
                 [0],
                 [0],
                 [0]])

s2 = np.dot(w2, y1) + b2
print(s2)

y2 = sigmoid(s2)
print(y2)

y = np.array([[1],
                    [1],
                    [0],
                    [0]])

de_dy2 = y2 - y
print(de_dy2)

dy2_ds2 = y2 * (1-y2)
print(dy2_ds2)

ds2_dw2 = y1
print(ds2_dw2)

de_dw2 = np.dot(de_dy2 * dy2_ds2, ds2_dw2.T)
print(de_dw2)

ds2_dy1 = w2
print(ds2_dy1) 

dy1_ds1 = y1 * (1-y1)
print(dy1_ds1)

ds1_dw1 = x
print(ds1_dw1)

de_dw1 = np.dot((np.dot((de_dy2 * dy2_ds2).T, ds2_dy1) * dy1_ds1.T).T, ds1_dw1.T)
print(de_dw1)


print(np.array([[1],[2],[3]]).reshape(-1,1))