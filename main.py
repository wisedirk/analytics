# https://towardsdatascience.com/deep-learning-with-python-neural-networks-complete-tutorial-6b53c0b06af0
# https://elitedatascience.com/learn-math-for-data-science


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from scipy.stats import norm


x = np.arange(1,10)
y = 3*x + 5

plt.plot(x,y)
plt.show()

x = tf.constant([[5, 2], [1, 3]])
print(x)

x = np.ones(shape=(3,1))
print(x)

x = np.linspace(-5, 5, 100)
y = norm.pdf(x)

plt.plot(x,y)
plt.show()

y = norm.cdf(x)
plt.plot(x,y)
plt.show()

print(norm.cdf(0))
print(norm.ppf(0.5), norm.ppf(0.25), norm.ppf(0.01))

x = np.arange(0,100)
y = norm.rvs(size=100)
plt.scatter(x,y)
plt.show()

print(y.mean(), y.std())