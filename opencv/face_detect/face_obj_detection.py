import cv2
face_cascade = cv2.CascadeClassifier("face_detect/haarcascade_frontalface_default.xml")
cap =cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detect_faces = face_cascade.detectMultiScale(gray, 1.1, 5)


    """
    detectMultiScale - scan and detect face
    scale factor = 1.1 (face kitna zoom krke dekh rhe ho)
    10% smaller -> 10% se kam hota jayega bounding box face detection ke liye
    minNeighbors = 5 (kitne aap ko strong clues chaiye ki yhi ye chehra hai) 5= safe checking, 3 = loose checking, 6 or more -> strict checking may not identify small faces
    """

    for (x,y,w,h) in detect_faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
    cv2. imshow("webcam face detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    


"""
x,y - top=left corner
(x+w, y+h) - bottom=right face capture krna h 
face = [
(100,150,80,80) face1
(250,120,90,90) face2
]
x - how far from left 
y - how far from top 
w - width of face 
h - height of face 
"""
