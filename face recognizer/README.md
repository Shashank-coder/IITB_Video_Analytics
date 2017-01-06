# IITB_Video_Analytics
This repository is for video analytics using Computer Vision, Machine Learning to summarize videos and obtain required results.


It mainly constitutes of three scripts-

- datasetcreator.py
	It detects faces using haarcascade and stores them in the dataset directory.

- trainer.py
	It uses opencv's LBPHFaceRecognizer and trains the machine id wise from the dataset then saves the trained model in the recognizer directory as .yml file.

- detector.py
	It uses face detection using haar cascade and predicts the id of the detected faces using the trained model. We can give each id a name then it will show the name
	corresponding to the id on the screen.

I haven't integrated the video summarization code with this script yet.


# USAGE

run the python scripts as usual in the order as above from the terminal in the face recognition directory.

python datasetcreator.py

python trainer.py

python detector.py
