# exit using escape
import cv2
import numpy as np

# haarcascade_frontalface_alt.xml works the best with good accuracy among all other frontal face cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)

id = raw_input("enter user id")
sample_num = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 1)
    # returns initial coordinates and width and height of faces
    for (x,y,w,h) in faces:
        sample_num = sample_num + 1
        cv2.imwrite("dataset/User."+str(id)+"."+str(sample_num)+".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.waitKey(100)

    cv2.imshow('img', img)
    cv2.waitKey(1)
    if(sample_num>20):
        break

cap.release()
cv2.destroyAllWindows()
