'''
1. 2 image combine
2. a part of image cut out 
3. flip/remove specific area  

cv2.bitwise_and(img1, img2) -> cut out the shape from a nother images
cv2.bitwise_or(img1, img2) -> combine multiple images in one image 
cv2.bitwise_not(img1) -> converting img from black to white or white to black 
'''
# img1 img2 height width same
# use only black and white image
# we can cut one particular image like circle from another image contain same shape

import cv2
import numpy as np 

img1 = np.zeros((300,300), dtype="uint8")
img2 = np.zeros((300,300), dtype="uint8")

cv2.circle(img1, (150,150), 100, 255, -1)
cv2.rectangle(img2, (100,100), (250,250), 255, -1)

bitwise_and = cv2.bitwise_and(img1, img2)
bitwise_or = cv2.bitwise_or(img1, img2)
bitwise_not = cv2.bitwise_not(img1)

cv2.imshow("circle", img1)
cv2.imshow("rectangle", img2)
cv2.imshow("AND", bitwise_and)
cv2.imshow("OR", bitwise_or)
cv2.imshow("NOT", bitwise_not)

cv2.waitKey(0)
cv2.destroyAllWindows()