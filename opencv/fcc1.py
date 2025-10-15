import cv2 as cv 
# reading image
'''
if image is not None:
    cv.imshow("image showing", image)
    cv.waitKey(0)
    cv.destroyAllWindows()
else: 
    print("image loaded successfully")
image = cv.imread("car_img.jpg")
cv.imshow('car', image)
cv.waitKey(0)  # waits for a specific delay, or time in ms for a key to be pressed
'''

#reading video
'''
capture = cv.VideoCapture('v1.mp4')
while True:
    isTrue,frame = capture.read() # we grab the video frame by frame. by utilising the captured read method, 
    cv.imshow('video', frame) # we display each frame of the video by using cv.imshow() mthd

    if cv.waitKey(20) & 0xFF==ord('d'): #if letter d pressed, the break out of the loop and stop displaying the video 
        break
capture.release()
cv.destroyAllWindows()
 
#output - it will display the video but after that it will give en error of -215 assertion failed. 
# it means that opencv could not find a media file at that particular location that you specified.
#reason why it is happened in the video is bcz the video ran out of frames opencv could not find any more frames after the last frames in this video.

'''
#resizing and resca;ling frames and images 
'''
import cv2
image = cv2.imread("car_img.jpg")
new_width = 300
new_height = 400

resized_image = cv2.resize(image, (new_width, new_height))

cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#resizing with a scaling factor 
'''
scale_down = 0.077
scale_up = 0.088

resize_down = cv.resize(image, None, fx=scale_down, fy = scale_down, interpolation=cv.INTER_LINEAR)
resize_up = cv.resize(image, None, fx=scale_up, fy = scale_up, interpolation=cv.INTER_LINEAR)

cv.imshow("original image", image)
cv.imshow("scaled down image", resize_down)
cv.imshow("scaled up image", resize_up)

cv.waitKey(0)
cv.destroyAllWindows()
'''

'''
fx = 0.088
fy = 0.066

resized_area = cv.resize(image, None, fx=fx, fy=fy, interpolation=cv.INTER_AREA)
resized_linear = cv.resize(image, None, fx=fx, fy=fy, interpolation=cv.INTER_LINEAR)
resized_cubic = cv.resize(image, None, fx=fx, fy=fy, interpolation=cv.INTER_CUBIC)
resized_nearest = cv.resize(image, None, fx=fx, fy=fy, interpolation=cv.INTER_NEAREST)

cv.imshow("original image", image)
cv.imshow("resized image with area", image)
cv.imshow("resized image with linear", image)
cv.imshow("resized image with cubic", image)
cv.imshow("resized image with nearest", image)
cv.waitKey(0)
cv.destroyAllWindows()
'''

'''
import cv2
# from google.colab.patches import cv2_imshow 

image = cv2.imread('flower.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

cv2.imshow("original_img", image)
cv2.imshow("rgb img",image_rgb)
cv2.imshow("hsv img",image_hsv)
cv2.imshow("gray img",image_gray)
cv2.imshow("lab img",image_lab)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
import cv2
import numpy as np

image = cv2.imread('flower.jpg')
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(image_hsv, lower_red, upper_red)
result = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("hsv_img", image_hsv)
cv2.imshow("img",result)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

# [255,0,0] -> pure red
# [0,255,0] -> pure green
# [0,0,255] -> pure blue

# jpg -> file ka size bohot small hoga
# png -> high quality image
# bmp -> uncompressed hoga, raw pixels honge jo ki harware me use hote h 
# tif -> super high resolution 

#how to load image
''' 
image = cv.imread("flower.jpg", flags=-1) # flag is optional -> 1 = color, 0 = grayscale, -1 = unchangable

# to display image 
cv.imshow("img", image)
cv.waitKey(0)
cv.destroyAllWindows()
'''

# saving image
'''
image = cv.imread("flower.jpg")
if image is not None:
    success = cv.imwrite("output_img.png", image)
    if success:
        print("image saved successfully as output_img.jpg")
    else:
        print("failed to save an image")   

else:
    print("error : could not load image") 
'''

# finding image dimension -> height, width, channel=3 (BGR) [if channel is missing it means image is in grayscale]
'''
image = cv.imread("car_img.jpg")
if image is not None:
    h, w, c = image.shape
    print(f"image loaded: \n Height: {h}\nwidth:{w}\nchannels: {c}")

else:
    print("could not load image")
'''

#grayscale conversion
'''
image = cv.imread("car_img.jpg") 
if image is not None:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) #detection will be working well when the img is in gray scale. if image is in grayscale then processing time and complexity will be less
    cv.imshow("grayscale image", gray)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print('could not load the image')
'''


