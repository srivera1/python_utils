import matplotlib.pyplot as plt
from sympy import *
x = Symbol('x')

f=sin(2*x)/(x)

def minGradientSearch(f,cur_x):
	gamma = 0.01
	precision = 0.00001
	previous_step_size = 1 
	max_iters = 10000
	iters = 0

	df = diff(f,x)
	error = []

	while (previous_step_size > precision) & (iters < max_iters):
	    prev_x = cur_x

	    # df/dx
	    cur_x -= gamma * df.evalf(subs={x:prev_x})

	    previous_step_size = abs(cur_x - prev_x)
	    iters+=1
	    error.append(previous_step_size)

	print( df.evalf(subs={x:cur_x}) )

	return cur_x,error

cur_x,error = minGradientSearch(f,.1)

print("Local minimum: " + str (cur_x) )

plt.plot(error)
plt.show()
