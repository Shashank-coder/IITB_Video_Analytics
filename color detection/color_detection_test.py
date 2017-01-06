# USAGE
# python save_key_events.py --output output

# import the necessary packages
from IITB_Video_Analytics.summary.keyclipwriter import KeyClipWriter
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-p", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=20,
	help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",
	help="codec of output video")
ap.add_argument("-b", "--buffer-size", type=int, default=32,
	help="buffer size of video clip writer")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to
# warmup
print("[INFO] warming up camera...")
# vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
# vs = cv2.VideoCapture("Horse Colourful Colour Song For Children Rhymes -- Animated Nursery Rhymes.mp4")
vs = cv2.VideoCapture("IITB_Video_Analytics/Videos/3D Horse Colors Songs Collection -- Horse Colorful Color Song For Children Rhymes.mp4")
time.sleep(2.0)

col = raw_input("Which Colour you want to detect? ")

# define the lower and upper boundaries of the "green" ball in
# the HSV color space
greenLower = (35, 50, 190)
greenUpper = (75, 255, 255)

blueLower = (75,50,190)
blueUpper = (150,255,255)

yellowLower = (25,50,190)
yellowUpper = (35,255,255)

redLower1 = (0, 50, 190)
redUpper1 = (15, 255, 255)

redLower2 = (175, 50, 190)
redUpper2 = (180, 255, 255)

# initialize key clip writer and the consecutive number of
# frames that have *not* contained any action
kcw = KeyClipWriter(bufSize=args["buffer_size"])
consecFrames = 0

# keep looping
while True:
	# grab the current frame, resize it, and initialize a
	# boolean used to indicate if the consecutive frames
	# counter should be updated
	ret, frame = vs.read()
	frame = imutils.resize(frame, width=600)
	updateConsecFrames = True

	# blur the frame and convert it to the HSV color space
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask1 = cv2.inRange(hsv, greenLower, greenUpper)
	mask1 = cv2.erode(mask1, None, iterations=2)
	mask1 = cv2.dilate(mask1, None, iterations=2)
	mask2 = cv2.inRange(hsv, blueLower, blueUpper)
	mask2 = cv2.erode(mask2, None, iterations=2)
	mask2 = cv2.dilate(mask2, None, iterations=2)
	mask3 = cv2.inRange(hsv, yellowLower, yellowUpper)
	mask3 = cv2.erode(mask3, None, iterations=2)
	mask3 = cv2.dilate(mask3, None, iterations=2)
	mask4 = cv2.inRange(hsv, redLower1, redUpper1)
	mask4 = cv2.erode(mask4, None, iterations=2)
	mask4 = cv2.dilate(mask4, None, iterations=2)
	mask5 = cv2.inRange(hsv, redLower2, redUpper2)
	mask5 = cv2.erode(mask5, None, iterations=2)
	mask5 = cv2.dilate(mask5, None, iterations=2)

	# find contours in the mask
	if(col == "green"):
		cnts = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL,
			  cv2.CHAIN_APPROX_SIMPLE)

	elif(col == "blue"):
		cnts = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL,
			  cv2.CHAIN_APPROX_SIMPLE)

	else:
		cnts = cv2.findContours(mask3.copy(), cv2.RETR_EXTERNAL,
			  cv2.CHAIN_APPROX_SIMPLE)

	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use it
		# to compute the minimum enclosing circle
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		updateConsecFrames = radius <= 10

		# only proceed if the redius meets a minimum size
		if radius > 10:
			# reset the number of consecutive frames with
			# *no* action to zero and draw the circle
			# surrounding the object
			consecFrames = 0
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 0, 255), 2)

			# if we are not already recording, start recording
			if not kcw.recording:
				timestamp = datetime.datetime.now()
				p = "{}/{}.avi".format(args["output"],
					timestamp.strftime("%Y%m%d-%H%M%S"))
				kcw.start(p, cv2.VideoWriter_fourcc(*args["codec"]),
					args["fps"])

	# otherwise, no action has taken place in this frame, so
	# increment the number of consecutive frames that contain
	# no action
	if updateConsecFrames:
		consecFrames += 1

	# update the key frame clip buffer
	kcw.update(frame)

	# if we are recording and reached a threshold on consecutive
	# number of frames with no action, stop recording the clip
	if kcw.recording and consecFrames == args["buffer_size"]:
		kcw.finish()

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# if we are in the middle of recording a clip, wrap it up
if kcw.recording:
	kcw.finish()

# do a bit of cleanup
cv2.destroyAllWindows()
# vs.stop()
