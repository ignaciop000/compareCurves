import os

def fileExists(filename):
	return os.path.isfile(filename)

def getCurveForImage(image):
	tmp = image.copy()
	if (tmp.dtype == cv2.CV_8UC3):
		gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
	elif (tmp.dtype == cv2.CV_8UC1):
		gray = tmp
	else:
		raise Exception('Unsupported image format')

	gray = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

	(_, contours, _) = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if len(contours) <= 0:
		return

	upperCurve = contours[0]
	if (len(upperCurve) <= 50):
		return	
