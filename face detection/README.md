# IITB_Video_Analytics
This repository is for video analytics using Computer Vision, Machine Learning to summarize videos and obtain required results.


This script is for frontal face detection. I have used haarcascade_frontalface_alt.xml here with a scale of 1.2, 2.

To process the haarcascade I have used opencv CascadeClassifier.

I have modified the code and integrated it with video summarization code such that whenever any face is detected it starts recording the video and stops when no face is detected.

# USAGE

face_detection.py

- After the repository is setup, go to terminal and goto the base folder and type the following command

python -m IITB_Video_Analytics.face\ detection.face_detection -o IITB_Video_Analytics/face\ detection/output

- After running this program your script will run

- To quit just press 'q'.

webcam_face_detection.py

- After the repository is setup, go to terminal and goto the base folder and type the following command

python -m IITB_Video_Analytics.face\ detection.webcam_face_detection -o IITB_Video_Analytics/face\ detection/output

- After running this program your script will run

- To quit just press 'q'.
