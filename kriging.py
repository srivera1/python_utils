import numpy as np
import datetime

""" kriging interpolation with
	numpy
"""

def kriging(x,y,z,values,x1,y1,z1):
	"""

	kriging interpolation with numpy
	
	interpolatedValues = kriging(x,y,z,values,x1,y1,z1)

		inputs - 7 arrays: 
				x[], y[], z[]:    3D location

				values[]: 		  some variable at
								  (x[],y[],z[])

				x1[], y1[], z1[]: locations to calculate
								  interpolation
		output - 1 array:
				 interpolatedValues: intperpolated values

	"""
	# Sampled Data Array
	ptos=np.append([x],[y],axis=0)
	ptos=np.append(ptos,[z],axis=0)

	# Regular mesh poins Array
	ptos1=np.append([x1],[y1],axis=0)
	ptos1=np.append(ptos1,[z1],axis=0)

	# relative distances betweem sampled points
	distances=np.matrix(x).T*np.matrix(x)*0
	for i in range(ptos.shape[1]): # Bottleneck1!
	    for j in range(ptos.shape[1]):
	        distances[i,j]=np.sqrt( np.sum(np.power(ptos[:,i]-ptos[:,j],2)) )

	# relative distance between sampled points and 
	# the new regular mesh
	distances1=np.matrix(x1).T*np.matrix(x)*0
	for i in range(ptos1.shape[1]): # Bottleneck2!
	    for j in range(ptos.shape[1]):
	        distances1[i,j]=np.sqrt(np.sum(np.power(ptos1[:,i]-ptos[:,j],2)))

	# kriging magic comes:
	return ((distances.I*distances1.T).T)*values.T


def vecPointsToMatrix(x,y,z,resX,resY):
	"""
	
		vecPointsToMatrix(x,y,z,resX,resY)
		2D mesh from 3D vectors with resolution
		resX and resY
	
	"""

	i=np.linspace(min(x),max(x),resX)
	j=np.linspace(min(y),max(y),resY)
	matrix=np.matrix(i).T*np.matrix(j)*0

	k=0
	for (m,l), value in np.ndenumerate(z):
		k=max(l,m)
		matrix[np.argmin(abs(i-x[k])),np.argmin(abs(j-y[k]))]=value
		k+=1

	return matrix
