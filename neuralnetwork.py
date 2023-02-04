# https://www.python-course.eu/neural_networks_backpropagation.php

import numpy as np
import scipy.special as sp

class NeuralNetwork:
    def __init__(self, nx, nh, ny):
        self.nx = nx
        self.nh = nh 
        self.ny = ny
        self.w1 = 2*np.random.random_sample((self.nh, self.nx)) - 1  
        self.b1 = 2*np.random.random_sample((self.nh, 1)) - 1  
        self.w2 = 2*np.random.random_sample((self.ny, self.nh)) - 1    
        self.b2 = 2*np.random.random_sample((self.ny, 1)) - 1
    
    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def der_sigmoid(self, x):
        return x * (1.0 - x)

 
    def feed_forward(self, x):
        z1 = self.w1 @ x + self.b1
        y1 = self.sigmoid(z1)

        z2 = self.w2 @ y1 + self.b2
        y2 = self.sigmoid(z2)

        return y1, y2

    def propagate_back(self, x, y1, y2, y):
        de_dy2 = y2 - y
        dy2_dz2 = y2 * (1-y2)
        dz2_dw2 = y1
        de_dw2 = (de_dy2 * dy2_dz2) @ dz2_dw2.T

        dz2_dy1 = self.w2
        dy1_dz1 = self.der_sigmoid(y1)
        dz1_dw1 = x
        de_dw1 = (((de_dy2 * dy2_dz2).T @ dz2_dy1) * dy1_dz1.T).T @ dz1_dw1.T
        
        return de_dw1, de_dw2

    def predict(self, x):
        x = np.array(x).reshape(-1, 1)
        return self.feed_forward(x)[1]
   
    def train_one(self, x, y, iter, rate):
        for i in range(iter):
            y1, y2 = self.feed_forward(x)
            de_dw1, de_dw2 = self.propagate_back(x, y1, y2, y)
            self.w1 -= rate * de_dw1
            self.w2 -= rate * de_dw2

    def train(self, data, iter, rate):
        for i in range(iter):
            for x, y in data:
                x = np.array(x).reshape(-1,1)
                y = np.array(y).reshape(-1,1)
                self.train_one(x, y, 1, rate)

    
if __name__ == "__main__":
    data = [
            ([0,0,0,0,0], [0,1]),
            ([0,0,0,0,1], [1,0]),
            ([0,0,0,1,0], [0,1]),
            ([0,0,0,1,1], [1,0]),
            ([0,0,1,0,0], [0,1]),
            ([0,0,1,0,1], [1,0]),
            ([0,0,1,1,0], [0,1]),
            ([0,0,1,1,1], [1,0]),
            ([0,1,0,0,0], [0,1]),
            ]
            
    
    
    nn = NeuralNetwork(5, 7, 2)
    
    nn.train(data, 10000, 1.0)

    print(nn.predict([0,0,0,0,1]))
    print(nn.predict([0,0,1,1,1]))
    print(nn.predict([0,0,1,0,0]))
    print(nn.predict([0,1,0,0,0]))
    print(nn.predict([0,1,0,0,1]))
    print(nn.predict([1,0,0,1,1]))
    print(nn.predict([1,0,0,0,0]))

