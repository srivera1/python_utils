from sympy import *
import sympy
from math import pi,e
from math import pow
import numpy as np

# variables
x = Symbol('x')
y = Symbol('y')
z = Symbol('z')

# model
G = sympy.Matrix([3*x-cos(y*z)-3./2.,
                  4.*x**2-625.*y**2+2.*y-1.,
                  sympy.exp(-x*y)+20.*z+(10.*pi-3.)/3.])

# Objective function
F     = G.T*G/2

# Objective functio gradient
gradF = G.jacobian([x,y,z]).T*G

# Step size
gamma = 0.001
precision = 0.00001
objective_function_eval = 1
# Max iteration number
max_iters = 190
iters = 0

# Initial location
cur_x=np.matrix([0.,0.,0.])
prev_x=np.matrix([0.,0.,0.])

while ( objective_function_eval < 100000    ) & \
      ( objective_function_eval > precision ) & \
      ( iters < max_iters ):

    prev_x[:] = cur_x[:]
    
    # Gradient of the model:
    dF=gamma * gradF.subs([(x,prev_x[0,0]),(y,prev_x[0,1]),(z,prev_x[0,2])])
    
    # Place the next point in the gradient descent direction
    for i,val in enumerate(dF):
        cur_x[0,i] -= val
    
    # Evaluaton the objective function at this value
    objective_function_eval = (F.subs([ (x,cur_x[0,0]), (y,cur_x[0,1]), (z,cur_x[0,2]) ]))[0,0]
    
    print('-----------------')
    print('Iteration: ' + str(iters))
    print('dF(x1): '    + str(dF))
    print('x0:     '    + str(prev_x))
    print('x1:     '    + str(cur_x))
    print('Objective function evaluation: ' + str(objective_function_eval))

    iters+=1


