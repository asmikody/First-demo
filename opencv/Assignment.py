# ASSIGNMENT1 

'''
#here i load the image then convert it into gray image after that display the image and then saved the image 
import cv2 as cv 
image = cv.imread("nature.jpg") # load the image 
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # converted into grayscale image
cv.imshow("nature image", image) # display the image
saved = cv.imwrite("output_image.jpg", gray) # saved the image
if saved:
    print("success")
else:
    print("failed")
cv.waitKey(0)
cv.destroyAllWindows()

'''
# ASSIGNMENT 1 CORRECTED ONE 
'''
import cv2 as cv 
image = cv.imread(input("enter image path: "))
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
choice = input("choose show image or save image ")
if choice == "show":
    cv.imshow("nature", gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

else:
    success = cv.imwrite(input("enter output name: "), gray)
    if success:
        print("success")
    else:
        print("failed")
        '''

# ASSIGNMENT 2
import cv2 as cv 
image = cv.imread(input("enter image path: "))
choice = input("enter your choice from line, circer, rectangle, text: ")
if choice.lower() == "line":
    pt1 = input("enter point 1 coordinates: ")
    pt2 = input("enter point 2 coordinates: ")
    cv.line(image, pt1, pt2, (255,0,0), 5)
elif choice.lower() == "circle":
    cv.circle(image, (50,50), 50, (0,255,255), -1)
elif choice.lower() == "rectangle":
    cv.rectangle(image, (50,50), (250,250), (255,0,255), 3)
elif choice.lower() == "text":
    cv.putText(image, "hello python programmer", (50,50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 5)



