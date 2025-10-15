# for open the camera 
''' 
import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() #read func return two things true/false and frames
    if not ret:
        print("could not read frame")
        break

    cv2.imshow("webcam feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("quitting...")
        break

cap.release() #it will close the camera 
cv2.destroyAllWindows()
'''

# videoWriter function -> we can save video to our laptop 
# cv2.VideoWriter(filename->(avi(audio video interleaf), mp4), codec, fps, frame_size)
# codec -> compression format

'''
import cv2 as cv

camera = cv.VideoCapture(0)

frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))

codec = cv.VideoWriter_fourcc(*'DIVX')
recorded = cv.VideoWriter("my_video.avi", codec, 20, (frame_width, frame_height))

while True:
    success, image = camera.read()
    if not success:
        break

    recorded.write(image)
    cv.imshow("Recording live", image)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# release AFTER loop
camera.release()
recorded.release()
cv.destroyAllWindows()
'''

