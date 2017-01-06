import imutils
import cv2
import numpy as np

cap = cv2.VideoCapture("IITB_Video_Analytics/Videos/3D Horse Colors Songs Collection -- Horse Colorful Color Song For Children Rhymes.mp4")

while cap.isOpened():
    _, img = cap.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # defining the range of yellow Color
    yellow_lower = np.array([25,50,190], np.uint8)
    yellow_upper = np.array([35,255,255], np.uint8)

    # defing the range of blue Color
    blue_lower = np.array([75,50,190], np.uint8)
    blue_upper = np.array([150,255,255], np.uint8)

    # including the range of blue in the image
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Morphological Transformation, Dilation
    kernal = np.ones((5, 5), "uint8")

    yellow = cv2.dilate(yellow, kernal)
    res = cv2.bitwise_and(img, img, mask = yellow)

    blue = cv2.dilate(blue, kernal)
    res1 = cv2.bitwise_and(img, img, mask = blue)

    # Tracking the yellow color
    (_,contours, heirarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
            cv2.putText(img, "yellow Color", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0))


    # Tracking the blue color
    (_,contours, heirarchy) = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area>300:
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(img, "Blue Color", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))


	# show the frame
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
