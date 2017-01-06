# exit using q
import cv2
import numpy as np

# haarcascade_frontalface_alt.xml works the best with good accuracy among all other frontal face cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)
rec = cv2.createLBPHFaceRecognizer()
rec.load("recognizer\\training.yml")
id = 0
font = cv2.cv.InintFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX__SMALL,5,1,0,4)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 1)
    # returns initial coordinates and width and height of faces
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        id, conf = rec.predict(gray[y:y+h, x:x+w])
        if(id==1):
            id = "Shashank"
        cv2.cv.PutText(cv2.cv.fromarray(img), str(id), (x,y+h), font, 255)

    cv2.imshow('img', img)

    if(cv2.waitKey(1) == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
