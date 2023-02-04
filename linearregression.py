# Gradient descent algorithm for linear regression
# https://medium.com/meta-design-ideas/linear-regression-by-using-gradient-descent-algorithm-your-first-step-towards-machine-learning-a9b9c0ec41b1

import numpy as np
import matplotlib.pyplot as plt

def plot(b,m,line):
	x = np.arange(0, 11, 10)
	y = m*x + b
	line.set_xdata(x)
	line.set_ydata(y)
	plt.draw()
	plt.pause(.01)
	

# minimize the "sum of squared errors". 
#This is how we calculate and correct our error
def compute_error_for_line_given_points(b,m,points):
	totalError = 0 	#sum of square error formula
	for i in range (0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		totalError += (y-(m*x + b)) ** 2
	return totalError/ float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
	#gradient descent
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		b_gradient += -(2/N) * (y - (m_current * x + b_current))
		m_gradient += -(2/N) * x * (y - (m_current * x + b_current))
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient) 
	return [new_b,new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations, line):
	b = starting_b
	m = starting_m
	for i in range(num_iterations):
		b,m = step_gradient(b, m, points, learning_rate)
		plot(b,m,line)
		print("After {0} iterations b = {1}, m = {2}, error = {3}".format(i, b, m, compute_error_for_line_given_points(b, m, points)))
	return [b,m]

def run():
	#Step 1: Collect the data
	points = np.array([[1,9.5],[2,7.8],[3,7.1],[4,6.3],[5,4.8],[6,4.1],[7,2.9],[8,2],[9,1]])
	learning_rate = 0.02 #how fast the data converge
	initial_b = 0 # initial y-intercept guess
	initial_m = 0 # initial slope guess
	num_iterations = 1000
	
	plt.rcParams['toolbar'] = 'None' 
	plt.figure("Linear Regression")
	plt.axis([0,10,0,10])
	plt.scatter(points.T[0], points.T[1])
	x = np.arange(0, 10, 1)
	y = initial_m*x + initial_b
	line, = plt.plot(x, y)
	plt.ion()
	plt.show()
	
	print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
	print("Running...")
		
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, line)
	
	print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

	plt.ioff()
	plt.show()	

# main function
if __name__ == "__main__":
	run()