# based on brightness we can convert image into black and white 

import cv2 
img = cv2.imread("nature.jpg", cv2.IMREAD_GRAYSCALE)
ret, thresh_img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
#if any pixel which is 120 above will be completely white and if it is less than 120 than it will be comletely black 
cv2.imshow("original image", img)
cv2.imshow("threshold image", thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
90 - 0 black 
130 - 255 white 
180 - 255 white
50 - 0 black

100, 120, 150 
best
'''


