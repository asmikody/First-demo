#Resizing and scaling
'''
import cv2 as cv
# syntax -> resized = cv.resize(src, dsize, fx, fy, interpolation)

image = cv.imread("car_img.jpg")
if image is None:
    print("image not found")
else:
    print("image loaded")
    # pass the dimesion of the image as wight first then height
    resized = cv.resize(image, (300,300))
    cv.imshow("original image", image)
    cv.imshow("resized image", resized)

    cv.imwrite("resized_output.jpg", resized)
    cv.waitKey(0)
    cv.destroyAllWindows()
'''

#cropping image using slicing in opencv 

# syntax -> cropped_img = image[startY:endY, startX:endX]
# here x is columns(left to right) and y is rows (top to bottom)

'''
import cv2 as cv
image = cv.imread("flower.jpg")
if image is not None:
    cropped = image[100:200, 50:150]
    cv.imshow("original", image)
    cv.imshow("cropped", cropped)
    cv.waitKey(0)
    cv.destroyAllWindows()
'''
# image rotation and flipping 
# syntax:-
# M = cv.getRotationMatrix2D(center, angle, scale)
# rotated_img = cv.wrapAffine(image, M, (width, height))

'''
import cv2 as cv
image = cv.imread("flower.jpg")
if image is None:
    print("could not load img")
else:
    (h,w) =image.shape[:2]

    center = (w//2, h//2)
    M = cv.getRotationMatrix2D(center, 90, 1.0)
    rotated = cv.warpAffine(image, M, (w,h))

    cv.imshow("original", image)
    cv.imshow("rotated 90 degree", rotated)
    cv.waitKey(0)
    cv.destroyAllWindows()
'''

#filiping image vertically, horizontally, and both
import cv2 as cv 
image = cv.imread("car_img.jpg")
if image is None:
    print("could not load image")
else:
    flipped_horizontal = cv.flip(image, 1)
    flipped_vertical = cv.flip(image, 0)
    flipped_both = cv.flip(image, -1)
     
    cv.imshow("original", image)
    cv.imshow("flipped_horizontal", flipped_horizontal)
    cv.imshow("flipped_vertical", flipped_vertical)
    cv.imshow("flipped_both", flipped_both)
    cv.waitKey(0)
    cv.destroyAllWindows()