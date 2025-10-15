import cv2
import numpy as np

img = cv2.imread(r"C:\Users\BAPS\Downloads\FINAL (3)\OpenCV\OpenCV\flower.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Mask for blue color (color thresholding)
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Threshold for brightness (grayscale thresholding)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

cv2.imshow("Mask (Color Based)", mask)
cv2.imshow("Threshold (Intensity Based)", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
