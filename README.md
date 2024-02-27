Card Recognition using OpenCV
=============================
This Python script performs real-time card recognition using OpenCV. It detects playing cards from a webcam feed, extracts their rank and suit, and attempts to identify them using template matching.

Dependencies
============
* `opencv-python` (cv2)
* `numpy`
* `os`
* `imutils`

How to Use
==========
1. Ensure you have a webcam connected to your system.
2. `pip install -r requirements.txt` to install dependencies to python interpreter.
3. Run the script.
4. Hold playing cards in front of the webcam.
5. The script will attempt to identify the rank and suit of each card in real-time.

Overview of the Script
======================

## Loading Card Images: 
The script loads pre-captured images of card ranks and suits to use as templates for comparison. see `Card_imgs` folder

## Webcam Initialization: 
Initialize the webcam feed for frame processing.

## Real-time Process:
Convert each frame to grayscale to match the images in our image folder.<br>
Determine the threshold level for binarization based on the background.<br>
Identify card contours in the frame.<br>
Extract the corner of the card containing the rank and suit.<br>
Process the rank and suit images for identification.<br>
Match the extracted rank and suit images with template images.<br>
Display the identified card rank and suit on the webcam feed.<br>

## Exiting the Script:
Press 'q' to quit the script.

Notes
=====
The script may require adjustments based on lighting conditions and camera placement for optimal performance.
Template matching accuracy may vary depending on the quality and variety of template images.

Author
======
This script was developed by Austin Lundberg.
