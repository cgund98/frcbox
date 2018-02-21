![alt text](https://github.com/cgund98/frcbox/raw/master/test.png "Test Image")
# Using Deep Learning for Game Piece Detection
Frcbox is my solution to a computer vision problem in the 2018 FIRST FRC robotics competition.  For the autonomous period of the match, the robot would need to be able to identify the box game piece in order to manipulate it.  To solve this problem I decided to use deep learning.  

A model was trained (finetuned) on 216 hand-collected and labeled images taken around the workplace of my robotics team, team 1747.  The model was built off of tensorflow's object detection API, which you can find the repository for [here](https://github.com/tensorflow/models/tree/master/research/object_detection).  

The files in this repo are for deploying onto a Pi or similar device.  If you are interested in the training of the model, that is in another repo. You can find that [here](https://github.com/cgund98/deeplearning/tree/master/tf-obj).

# Setup
## OS
As recently mentioned, this version was deployed on to the Raspberry Pi 3.  I recommend using the Raspbian OS. If you don't you will end up having to compile dependencies yourself.  That's a huge pain and takes forever on the Pi so just do yourself a favor and use Raspbian.

## Dependencies
Tensorflow and OpenCV are the main dependencies for this app.  

I wrote the script with OpenCV2 and there are plenty of online tutorials on how to install it onto the Pi.  

Normally you can use `pip` to install tensorflow by simply running `pip install tensorflow==1.4.1`, but unfortunately it will not have a recent enough version to work with the object detection model.  Instead of compiling from scratch, you can find a compiled file to install from [here](https://github.com/lhelontra/tensorflow-on-arm/releases).  To get the file on the Pi I simply used: 

`wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.4.1/tensorflow-1.4.1-cp35-none-linux_armv7l.whl'`

All you have to do to install it is `pip install tensorflow-1.4.1-cp35-none-linux_armv7l.whl` or from whichever file you downloaded.

#### ** It is important that you install tensorflow-1.4.1 or 1.5 as the model will not work if you use another version.

To check if the dependences are installed you can enter a python console with `python` or `python3`, whichever version you use, and then try:
```python
import tensorflow
import cv2
```
If it doesn't throw any errors you the dependencies should be installed.

# Running the script
Right now there are two primary scripts, `detect.py` and `pi_detect.py`.  `detect.py` is just a general detection script meant for testing on a file.  `pi_detect.py` is a live detection script that would be run when mounted to the robot.  Both scripts are full of unused functions and commented out lines.  Feel free to dig through the code to get what you need out of it.    

## detect.py
You should be able to test out the detection model by running:

`python detect.py ../images/test.jpg`

Replace `python` with `python3` if needed and replace `../images/test.jpg` to the path of your test image.

** You may have to change the line `PATH_TO_CKPT = 'output_inference_graph.pb/frozen_inference_graph.pb'` to `PATH_TO_CKPT = 'output_inference_graph-1.4.1.pb/frozen_inference_graph.pb'` if you installed tensorflow version 1.4.1 instead of 1.5.

A window should pop up with the image with all boxes shown in a colored box.  

** If you are running this from your pi via ssh and want to see the pop-up, you need to ssh into it with `ssh pi@your-ip -Y`.  The pop-up should be able to slowly show up on your client pc.  

## pi_detect.py
`python pi_detect.py` simply fetches frames from the Raspberry Pi camera and searches each frame for the game piece.  It outputs an array of coordinates of bounding boxes in the form `[x1, y1, x2, y2]`.  

# More information
Frcbox is a work in progress and surely has several issues.  If you have any questions, tips, or comments feel free to create an issue or get ahold of me at (gundlachcallum@gmail.com)[mailto:gundlachcallum@gmail.com].  
