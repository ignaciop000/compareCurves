import matplotlib.pyplot as pl
import numpy as np
import cv2

"""
1st and 2nd derivative of 1D gaussian 
"""
def getGaussianDerivs(sigma, M):
	L = int((M - 1) / 2)
	sigma_sq = sigma * sigma
	sigma_quad = sigma_sq * sigma_sq
	gaussian = np.zeros(M)
	gaussian1D = np.zeros(M)
	gaussian2D = np.zeros(M)

	g = cv2.getGaussianKernel(M, sigma, cv2.CV_64F)    
	for i in range(-L,L+1):
		idx = int(i+L)
		gaussian[idx] = g[idx]
		gaussian1D[idx] = (-i/sigma_sq) * g[idx]
		gaussian2D[idx] = (-sigma_sq + i*i)/sigma_quad * g[idx]   
	return (gaussian, gaussian1D, gaussian2D)

def polyLineSplit(curve):    
	curveX,curveY = zip(*curve)
	return (curveX, curveY)

def polyLineMerge(curveX, curveY):
	return zip(curveX, curveY)

def f(x):
	return np.sin(x) + np.random.normal(scale=0.1, size=len(x))

"""
1st and 2nd derivative of smoothed curve point
"""
def getdX(curveX, n, sigma, gaussian, gaussian1D, gaussian2D, isOpen):
	L = (len(gaussian) - 1) / 2;
	gx = 0
	dgx = 0
	d2gx = 0
	for k in range(-L, L+1):
		if (n-k < 0):
			if isOpen:
				#open curve - mirror values on border
				x_n_k = curveX[-(n-k)]
			else:
				#closed curve - take values from end of curve
				x_n_k = curveX[len(curveX)+(n-k)]
		elif n-k > len(curveX)-1:
			if isOpen:
				#mirror value on border
				x_n_k = curveX[n+k] 
			else:
				x_n_k = curveX[(n-k)-(len(curveX))]
		else:
			x_n_k = curveX[n-k];

		gx += x_n_k * gaussian[k + L] #gaussians go [0 -> M-1]
		dgx += x_n_k * gaussian1D[k + L]
		d2gx += x_n_k * gaussian2D[k + L]
	return (gx, dgx, d2gx)

"""
0th, 1st and 2nd derivatives of whole smoothed curve
"""
def getdXcurve(curveX, sigma, gaussian, gaussian1D, gaussian2D, isOpen):

	gx = np.zeros(len(curveX))
	dx = np.zeros(len(curveX))
	d2x = np.zeros(len(curveX))
	for i in range(0, len(curveX)):
		gx[i], dx[i], d2x[i] = getdX(curveX,i,sigma,gaussian,gaussian1D,gaussian2D,isOpen)
	return (gx, dx, d2x)

def resampleCurveXY(curveX, curveY, N, isOpen):
	resamplepl = np.empty(N, dtype=object)
	resamplepl[0] = (curveX[0], curveY[0])
	pl = polyLineMerge(curveX,curveY)
	pl = np.asarray(pl)
	pl = np.float32(pl)
	pl_length = cv2.arcLength(pl, False)
	resample_size = pl_length / N
	curr = 0
	dist = 0.0
	i = 1
	while i<N:       
		last_dist = cv2.norm(pl[curr] - pl[curr+1])
		dist += last_dist
		if dist >= resample_size:
			#put a point on line
			_d = last_dist - (dist-resample_size)
			cp = (pl[curr][0],pl[curr][1])
			cp1 = (pl[curr+1][0],pl[curr+1][1])
			dirv = np.subtract(cp1, cp)
			
			dirv = dirv * (1.0 / cv2.norm(dirv))
			resamplepl[i] = cp + dirv * _d
			i += 1
			 
			dist = last_dist - _d #remaining dist         
			 
			#if remaining dist to next point needs more sampling... (within some epsilon)
			while dist - resample_size > 1e-3:
				resamplepl[i] = resamplepl[i-1] + dirv * resample_size
				dist -= resample_size
				i += 1
		curr+=1    
	 
	(resampleX,resampleY) = polyLineSplit(resamplepl)
	return (resampleX, resampleY)

def resampleCurve( curve, N, isOpen = False):	
	curveX, curveY = polyLineSplit(curve)
	resampleX,resampleY = resampleCurveXY(curveX,curveY,N,isOpen)
	curveResample = polyLineMerge(resampleX, resampleY)
	return curveResample

"""
	compute curvature of curve after gaussian smoothing 
	from "Shape similarity retrieval under affine transforms", Mokhtarian & Abbasi 2002
	curvex - x position of points
	curvey - y position of points
	kappa - curvature coeff for each point
	sigma - gaussian sigma
"""
def computeCurveCSS(curveX, curveY, sigma, isOpen):
	M = round((10.0*sigma+1.0) / 2.0) * 2 - 1
 
	g,dg,d2g = getGaussianDerivs(sigma,M)

	smoothX,X,XX = getdXcurve(curveX,sigma,g,dg,d2g, isOpen);
	smoothY,Y,YY = getdXcurve(curveY,sigma,g,dg,d2g, isOpen);
		 
	kappa = np.zeros(len(curvex))    
	for i in range(0, len(curvex)):
		# Mokhtarian 02' eqn (4)
		kappa[i] = (X[i]*YY[i] - XX[i]*Y[i]) / pow(X[i]*X[i] + Y[i]*Y[i], 1.5)
	return kappa

"""
	find zero crossings on curvature
"""
def findCSSInterestPoints(kappa):
	crossings = []
	for i in range(0, len(kappa)-1):
		if ((kappa[i] < 0 and kappa[i+1] > 0) or kappa[i] > 0 and kappa[i+1] < 0):
			crossings.append(i)
	return crossings

def convertCurve(curve):
	output = [(i[0],i[1]) for i in curve]	
	return output

if __name__ == "__main__":
	x = np.linspace(1, 50, 500)
	curve = zip(x, f(x))
	
	sigma = 1.0
	M = int((10.0*sigma+1.0) / 2.0) * 2 - 1
	#M is an odd number
	gaussian, gaussian1D, gaussian2D = getGaussianDerivs(sigma,M);
	curveX, curveY = polyLineSplit(curve)
	smoothX,X,XX = getdXcurve(curveX,sigma,gaussian,gaussian1D,gaussian2D, True);
	smoothY,Y,YY = getdXcurve(curveY,sigma,gaussian,gaussian1D,gaussian2D, True);
	smooth = polyLineMerge(smoothX,smoothY);
	resampleX, resampleY = resampleCurveXY(curveX, curveY, 50, True)
	pl.plot(curveX, curveY)
	pl.plot(smoothX, smoothY)
	pl.plot(resampleX, resampleY)
	pl.show()