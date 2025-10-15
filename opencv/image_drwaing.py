#drwaing a line- cv2.line()
# syntax
# cv2.line(img, pt1, pt2, color, thickness)
'''
import cv2 as cv 
image = cv.imread("icon.png")
if image is None:
    print("Oops! your image is not working")
else:
    print("image loaded successfully!")

    pt1 = (50, 100)
    pt2 = (300, 100)
    color = (0, 255, 0)
    thickness = 4

    cv.line(image, pt1, pt2, color, thickness)
    cv.imshow("Line Drawing", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

'''
# drawing a rectangle - syntax - cv2.rectangle(img, pt1, pt2, color, thickness)

'''
import cv2 as cv 
image = cv.imread("nature.jpg")
if image is None:
    print("Oops! your image is not working")
else:
    print("image loaded successfully!")
    pt1 = (50,50)
    pt2 = (150,200)
    color = (0,255,255)
    thickness = 3

    cv.rectangle(image, pt1, pt2,color, thickness)
    cv.imshow("image focusing rectangle", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
'''

# drawing a circle - cv.circle(img, center, radius, color, thickness)
'''
import cv2 as cv 
image = cv.imread("nature.jpg")
if image is None:
    print("Oops! your image is not working")
else:
    print("image loaded successfully!")
    cv.circle(image, (50,50), 50, (255,0,0), -1) #thickness = -1 means completely filled circle
    cv.imshow("Drawing circle", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
'''

# adding text- cv.putText() - syntax 
# cv.putText(img, text, org, font, fontScale, color, thickness)

'''
import cv2 as cv 
image = cv.imread("car_img.jpg")
if image is None:
    print("Oops! your image is not working")
else:
    print("image loaded successfully!")
    cv.putText(image, "hello python programmer",(50,50), cv.FONT_HERSHEY_SIMPLEX,  1.2, (0,255,255), 2)
    cv.imshow("image focusing rectangle", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
'''


