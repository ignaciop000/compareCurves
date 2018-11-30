def getCurveForImage(image):	
	tmp = image.copy()
	if (tmp.dtype == cv2.CV_8UC3):
		gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	elif (tmp.dtype == cv2.CV_8UC1):
		gray = tmp
	else:
		raise Exception('Unsupported image format')

	gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

	#cv2.imshow("input",gray);
	
	(_, contours, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) <= 0:
		return

	upperCurve = contours[0]
	if (len(upperCurve) <= 50):
		return
	
	#find minimal and maximal X coord
	x, y = polyLineSplit(contours[0])
	Point minxp,maxxp;
	_, _, minxp, maxxp = cv2.minMaxLoc(x)
	minx = minxp[0]
	maxx = maxxp[0]
	if minx > maxx:
		minx, maxx = maxx, minx
	
	#take lower and upper halves of the curve
	vector<Point> upper,lower;
	upper.insert(upper.begin(),contours[0].begin()+minx,contours[0].begin()+maxx);
	lower.insert(lower.begin(),contours[0].begin()+maxx,contours[0].end());
	lower.insert(lower.end(),contours[0].begin(),contours[0].begin()+minx);
	
	#test which is really the upper part, by looking at the y-coord of the mid point
	
	if (lower[lower.size()/2].y <= upper[upper.size()/2].y) {
		curve_upper = lower;
		curve_lower = upper;
	} else {
		curve_upper = upper;
		curve_lower = lower;
	}
	
	#make sure it goes left-to-right
	if (curve_upper.front().x > curve_upper.back().x) { //hmmm, need to flip
		reverse(curve_upper.begin(), curve_upper.end());
	}		
	
	whole.clear();
	whole.insert(whole.begin(),curve_upper.rbegin(),curve_upper.rend());
	whole.insert(whole.begin(),curve_lower.begin(),curve_lower.end());
	return (whole, curve_upper, curve_lower)

def getCurveForImage(image, onlyUpper, getLower):
	whole,upper,lower = getCurveForImage(image);
	if onlyUpper:
		if getLower:
			curve = lower
		else
			curve = upper
	else
		curve = whole
	return curve

if __name__ == "__main__":	
	#pl.plot(curveX, curveY)
	#pl.plot(smoothX, smoothY)
	#pl.plot(resampleX, resampleY)
	pl.show()

