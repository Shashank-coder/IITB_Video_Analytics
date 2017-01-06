# IITB_Video_Analytics
This repository is for video analytics using Computer Vision, Machine Learning to summarize videos and obtain required results.


This script is used to detect colors in a video.

color_detection_test.py

In this script I have converted the frames to hsv, then I have used opencv to create masks of individual colors and then found contours of the color asked since it is a query based script.
If a single contour is found in a frame then a minimum enclosing circle is drawn enclosing all the contours in that frame and if its radius is larger than 10 then it is reflected on the screen and recording is performed.

multiple_color_detection.py

In this script I have defined hsv range of some colors then looped over each found contours of each colour found and marked them with rectangle.
Video summarization is not available in this script.

# USAGE

color_detection_test.py

- After the repository is setup, go to terminal and goto the base folder and type the following command

python -m IITB_Video_Analytics.color\ detection.color_detection_test -o IITB_Video_Analytics/color\ detection/output

- After running this program your script will run

- To quit just press 'q'.


multiple_color_detection.py

- After the repository is setup, go to terminal and goto the base folder and type the following command

python -m IITB_Video_Analytics.color\ detection.multiple_color_detection

- After running this program your script will run

- To quit just press 'q'.

