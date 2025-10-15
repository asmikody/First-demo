import cv2
import numpy as np

# 1️ Read the image
img = cv2.imread(r"C:\Users\BAPS\Downloads\FINAL (3)\OpenCV\OpenCV\flower.jpg")   #  change to your image path
cv2.imshow("Original Image", img)

# 2️ Convert to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", hsv)

# 3️ Apply Gaussian Blur
blur = cv2.GaussianBlur(hsv, (5, 5), 0)
cv2.imshow("Blurred Image", blur)

# 4️ Define color ranges in HSV

# Blue color range
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

# Green color range
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# Yellow color range
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# 5️ Create masks for each color
mask_blue = cv2.inRange(blur, lower_blue, upper_blue)
mask_green = cv2.inRange(blur, lower_green, upper_green)
mask_yellow = cv2.inRange(blur, lower_yellow, upper_yellow)

cv2.imshow("Blue Mask", mask_blue)
cv2.imshow("Green Mask", mask_green)
cv2.imshow("Yellow Mask", mask_yellow)

# 6️ Combine all masks together
combined_mask = cv2.bitwise_or(mask_blue, mask_green)
combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)
cv2.imshow("Combined Mask", combined_mask)

# 7️ Find contours on combined mask
contours, hierarchy = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 8️ Draw contours on original image
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
cv2.imshow("Contours on Original", result)

# 9️ Optionally show each color-masked region separately
res_blue = cv2.bitwise_and(img, img, mask=mask_blue)
res_green = cv2.bitwise_and(img, img, mask=mask_green)
res_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)

cv2.imshow("Detected Blue", res_blue)
cv2.imshow("Detected Green", res_green)
cv2.imshow("Detected Yellow", res_yellow)

#  Wait and close
cv2.waitKey(0)
cv2.destroyAllWindows()




'''
cv2.GaussianBlur(img, (5,5), 0)
cv2.medianBlur(img, 5)
cv2.blur(img, (5,5))
cv2.bilateralFilter(img, 9, 75, 75)

'''



'''

import cv2

img = cv2.imread(r"C:\path\to\your\image.jpg")
if img is None:
    print("❌ Image not found.")
    exit()

# Apply Gaussian Blur
gaussian = cv2.GaussianBlur(img, (5,5), 0)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Gaussian Blur", gaussian)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''import cv2

img = cv2.imread(r"C:\path\to\your\image.jpg")
if img is None:
    print("❌ Image not found.")
    exit()

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply simple binary threshold
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Show results
cv2.imshow("Gray Image", gray)
cv2.imshow("Thresholded Image", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''


'''
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

'''

'''
import cv2

img = cv2.imread(r"C:\path\to\your\image.jpg")
if img is None:
    print("❌ Image not found.")
    exit()

# Convert to gray and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on copy of original image
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Contours", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''