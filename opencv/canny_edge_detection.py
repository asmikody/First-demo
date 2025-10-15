# edges =cv2.Canny(image, threshold1, threshold2)

# 1. detect border's
# 2. separate objects
# 3. feature extraction

# threshold if going above 150 then white kr do otherwise balck kr do 

import cv2
img = cv2.imread("flower.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    print("could no tload img")
    exit()
edges = cv2.Canny(img, 50, 150)
cv2.imshow("original image", img)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
90 - 0 black 
130 - 255 white 
180 - 255 white
50 - 0 black
'''


