# image filtering ->
# 1. blurring -> gaussin, median, bilateral, normal
# 2. noise
# 3. smoothing
# 4. kernel

# gaussian blur - image ko soft krna, removing noise, dust and sharp images, smoothing

# blurred_image = cv.GaussianBlur(image, (kernel_size_x, kernel_size_y))

# kernel size -(3,3)->light blur ; (9,9) -> strong blur; (21,21)-> super blur

'''
import cv2 
image = cv2.imread("envy.jpg")
blurred = cv2.GaussianBlur(image,(7,7), 0)

cv2.imshow("original image", image)
cv2.imshow("blurred image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#median blur 
# blurred = cv2.medianBlur(image, kernel_size)

'''
import cv2
image = cv2.imread("envy.jpg")
blurred = cv2.medianBlur(image, 5)

cv2.imshow("original", image)
cv2.imshow("clean image", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# sharpening -> highlight the edges, pixel ke bich me jo contrast hota h usko boost kr deta h
# cv2.filter(src, depth, kernel)

'''
import cv2
import numpy as np

image = cv2.imread("low_res.jpg")

sharpen_kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharpened = cv2.filter2D(image, -1, sharpen_kernel)
cv2.imshow("original image", image)
cv2.imshow("sharpened image", sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import cv2

img = cv2.imread("flower.jpg")

# Increase brightness by 50
bright_img = cv2.convertScaleAbs(img, alpha=1, beta=50)

# Decrease brightness by 50
dark_img = cv2.convertScaleAbs(img, alpha=1, beta=-50)

# Increase contrast (alpha > 1.0)
high_contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

# Decrease contrast (alpha < 1.0)
low_contrast = cv2.convertScaleAbs(img, alpha=0.5, beta=0)

blur = cv2.blur(img, (5,5))  # kernel size

gaussian = cv2.GaussianBlur(img, (5,5), 0)

median = cv2.medianBlur(img, 5)

bilateral = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow("Original", img)
# cv2.imshow("Brighter", bright_img)
# cv2.imshow("Darker", dark_img)
# cv2.imshow("high_contrast", high_contrast)
# cv2.imshow("low_contrast", low_contrast)
cv2.imshow("blur", blur)
cv2.imshow("guassian", gaussian)
cv2.imshow("median blur" ,median)
cv2.imshow("bilateral", bilateral)

cv2.waitKey(0)
cv2.destroyAllWindows()

