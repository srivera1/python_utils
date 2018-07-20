import numpy as np
import matplotlib.pyplot as plt

from kriging import kriging
from kriging import vecPointsToMatrix


def dataModel(x,y):
	# simplest non linear model
	return np.matrix(np.cos(x)+np.cos(y))

if __name__=="__main__":

	# 3D Sampled data points with some gaussian density:
	NSampledPoints=200
	x=np.random.normal(2,1,NSampledPoints)
	y=np.random.normal(2,2,NSampledPoints)
	z=np.random.normal(50,10,NSampledPoints)*0
	values=dataModel(x,y)
	
	# We expect to interpolate the model
	# values on a regular mesh.
	# Interpolation positions:
	NMeshPointsX=50 # Mesh with size 50x60
	NMeshPointsY=60 
	NMeshPointsZ=50 
	x1=np.linspace(np.min(x),np.max(x),NMeshPointsX)
	y1=np.linspace(np.min(y),np.max(y),NMeshPointsY)
	z1=np.linspace(np.min(z),np.max(z),NMeshPointsZ)

	x2=np.arange(0)
	y2=np.arange(0)
	z2=np.arange(0)
	for xi in x1:
		for yi in y1:
			#for zi in z:
			x2=np.append(x2,[xi],axis=0)
			y2=np.append(y2,[yi],axis=0)
			#z2=np.append(z2,[zi],axis=0)*0
	z2=x2*0

	# 2D grid of sampled points proyected over a regular mesh:
	c1=vecPointsToMatrix(x,y,values,NMeshPointsX,NMeshPointsY)

	# interpolation at the points on the regular mesh:
	krigingInterpolation=kriging(x,y,z,values,x2,y2,z2)
	# 2D matrix with the interpolated values:
	c2=vecPointsToMatrix(x2,y2,krigingInterpolation,NMeshPointsX,NMeshPointsY)

	# For comparison with the model,
	# 2D exact value on the interpolated locations of dataModel(x,y):
	c3=c2*0
	print(c3.shape)
	for i in range(x1.shape[0]):
		for j in range(y1.shape[0]):
			c3[i,j]=dataModel(np.array([x1[i]]),np.array([y1[j]]))[0,0]

	if True:
		fig=plt.figure()
		ax1 = fig.add_subplot(221)
		ax1.contour(c3,100)
		plt.title("Exact Model")
		ax2 = fig.add_subplot(222)
		ax2.contour(c1,100)
		plt.title("Sampled points")
		ax3 = fig.add_subplot(223)
		ax3.contour(c2,100)
		plt.title("Kriging interpolated")
		ax3 = fig.add_subplot(224)
		# larger error come at non sampled areas:
		ax3.contour(np.abs(c3-c2),100)
		plt.title("Error")
		plt.show()

