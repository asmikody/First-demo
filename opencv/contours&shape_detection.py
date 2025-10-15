# finding contours - use cv2.findContours to locate object boundaries 
# shape detection - approximate shapes using approxPloyDP for object recognition 
# Drawing Contours - highlight detected objects by drawing around their edges 

# contours, hierarchy = cv2.findContours(image, mode, method)

"""
import cv2
img = cv2.imread("triangle.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawCountours()
# cv2.drawContours(image, contours, contour_index, color, thickness )

cv2.drawContours(img, contours, -1, (0,255,0), 3)
cv2.imshow("contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# approx = cv2.approxPloyDP(contour, epsilon, True)

# 0.01 * arcLength

import cv2
img = cv2.imread("five.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cv2.drawCountours()
# cv2.drawContours(image, contours, contour_index, color, thickness)

cv2.drawContours(img, contours, -1, (0,255,0), 3)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    print(approx)
    corners = len(approx)
    print(corners)

    if corners == 3:
        shape_name = "Triangle"
    elif corners == 4:

        shape_name = "Rectangle"
    elif corners == 5:
        shape_name = "Pentagon"
    elif corners>5:
        shape_name = "Circle"
    else:
        shape_name = "unknown"

    cv2.drawContours(img, [approx], 0, (0,255,0), 2)
    x=approx.ravel()[0]
    y = approx.ravel()[1] - 10
    cv2.putText(img, shape_name, (x,y), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,0,0))

cv2.imshow("contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
 