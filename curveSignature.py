import cv2
import numpy as np
import curveCSS
import matplotlib.pyplot as pl

def getAllCurveForImage(image):
	tmp = image.copy()
	if (tmp.dtype == np.uint8):
		gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	elif (tmp.dtype == cv2.CV_8UC1):
		gray = tmp
	else:
		raise Exception('Unsupported image format')

	_,gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
	
	#cv2.imshow("input",gray)	
	
	(_, contours, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) <= 0:
		return	
	upperCurve = [(i[0][0],i[0][1]) for i in contours[0]]
	if (len(upperCurve) <= 50):
		return
	
	#find minimal and maximal X coord
	x, y = curveCSS.polyLineSplit(upperCurve)
	_, _, minxp, maxxp = cv2.minMaxLoc(x)
	minx = minxp[1]
	maxx = maxxp[1]

	if minx > maxx:
		minx, maxx = maxx, minx
	
	#take lower and upper halves of the curve
	upper = upperCurve[minx:maxx]
	lower = upperCurve[maxx:] + upperCurve[:minx]

	#test which is really the upper part, by looking at the y-coord of the mid point	
	if lower[len(lower)/2][1] <= upper[len(upper)/2][1]:
		curve_upper = lower
		curve_lower = upper
	else:
		curve_upper = upper
		curve_lower = lower
	
	#make sure it goes left-to-right
	if (curve_upper[0][0] > curve_upper[-1][0]): #hmmm, need to flip		
		curve_upper = curve_upper[::-1]
	
	whole = curve_upper[::-1] + curve_lower[:]
	return (whole, curve_upper, curve_lower)

def getCurveForImage(image, onlyUpper, getLower = False):
	whole,upper,lower = getAllCurveForImage(image);
	if onlyUpper:
		if getLower:
			curve = lower
		else:
			curve = upper
	else:
		curve = whole
	return curve

if __name__ == "__main__":	
	image = cv2.imread("camel-curve.png")
	#whole, curve_upper, curve_lower = getAllCurveForImage(image)
	#pl.plot(*zip(*whole))
	#pl.plot(*zip(*curve_upper))
	#pl.plot(*zip(*curve_lower))
	curve = getCurveForImage(image, False)
	pl.plot(*zip(*curve))
	curveResample = curveCSS.resampleCurve(curve, 200, False);
	pl.plot(*zip(*curveResample))
	a_p2d = curveCSS.convertCurve(curveResample)
	
	
	pl.show()

