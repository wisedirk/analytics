# https://www.tutorialspoint.com/matplotlib/matplotlib_pylab_module.htm
# https://medium.com/meta-design-ideas/linear-regression-by-using-gradient-descent-algorithm-your-first-step-towards-machine-learning-a9b9c0ec41b1

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def f(a,b,x):
	return a*x+b

def err(a,b,points):
	e = 0
	N = float(len(points))
	for p in points:
		x = p[0]
		y = p[1]
		e += (y - (a*x + b))**2
	return e/N

def step_gradient(a,b,points,learning_rate):
	da = 0
	db = 0
	N = float(len(points))
	for p in points:
		x = p[0]
		y = p[1]
		da += -(2/N) * x * (y - (a*x + b))
		db += -(2/N) * (y - (a*x + b))
	a -= (learning_rate * da) 
	b -= (learning_rate * db)
	return [a,b]

def gradient_descent(a,b,points,learning_rate,num_iterations):
	for i in range(num_iterations):
		plot(a,b,points,i)
		a,b = step_gradient(a,b,points,learning_rate)
	return [a,b]

def plot(a,b,points,i):
	e = err(a,b,points)
	t = 'i={0:d} a={1:+7.3f} b={2:+7.3f} e={3:7.3f}'.format(i,a,b,e)
	ax2.text(1,9,t,bbox = {'facecolor':'white', 'edgecolor':'white'})	
	
	ta.append(a)
	tb.append(b)
	te.append(e)
	path, = ax1.plot3D(ta, tb, te, 'r.-')

	x = np.array([0,10])
	y = f(a,b,x)
	line.set_data(x,y)

	plt.draw()
	plt.pause(0.001)
	 

ga = 0.5
gb = 1.0

x = np.linspace(1,10,10)
y = f(ga,gb,x)
y += np.random.rand(len(y)) - 0.5
points = np.stack((x,y), axis = -1)

x = np.array([0,10])
y = f(ga,gb,x)

a = np.linspace(-10, 10)
b = np.linspace(-10, 10)
A,B = np.meshgrid(a, b)
E = err(A,B,points)

ta = []
tb = []
te = []


plt.rcParams['toolbar'] = 'None' 
fig = plt.figure()

ax1 = fig.add_subplot(1,2,1, projection='3d')
ax1.plot_surface(A, B, E, cmap='viridis', edgecolor='none')
path, = ax1.plot3D(ta, tb, te, 'r.')
ax1.set_xlabel("a")
ax1.set_ylabel("b")
ax1.set_zlabel("error")
ax1.set_title('Gradient Descent')

ax2 = fig.add_subplot(1,2,2)
line, = ax2.plot(x,y)
ax2.scatter(points.T[0], points.T[1])
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_xlim(0,10)
ax2.set_ylim(0,10)
ax2.set_title('Linear Regression')

plt.ion()
plt.show()

a,b = gradient_descent(-10,-10,points,0.01,1000)
print(a,b)

plt.ioff()
plt.show()

