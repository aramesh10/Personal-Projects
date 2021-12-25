# This program can detect human and cat faces using OpenCV and already trained Haar Cascade classifiers
# Last modified by Aditya Ramesh

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier(r'haarcascade_frontalface_default.xml')
cat_cascade = cv.CascadeClassifier(r'haarcascade_frontalcatface_extended.xml')

print("Press q to quit...")

while True:
    ret, frame = cap.read()

    face = face_cascade.detectMultiScale(image=frame, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in face:
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cat = cat_cascade.detectMultiScale(image=frame, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in cat:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
 
    frame = cv.putText(frame, "Cat (green) | Human (blue)", (5,25), cv.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 2)

    cv.imshow('video', frame)
        

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
