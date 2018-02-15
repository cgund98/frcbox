# Using Deep Learning for Game Piece Detection
Frcbox is my solution to a computer vision problem in the 2018 FIRST FRC robotics competition.  For the autonomous period of the match, the robot would need to be able to identify the box game piece in order to manipulate it.  To solve this problem I decided to use deep learning.  

A model was trained (finetuned) on 216 hand-collected and labeled images taken around the workplace of my robotics team, team 1747.  The model was built off of tensorflow's object detection API, which you can find the repository for [here](https://github.com/tensorflow/models/tree/master/research/object_detection).  

The files in this repo are only for deploying onto a pi or similar device.  If you are interested in the training of the model, that is in another repo. You can find that [here](https://github.com/cgund98/deeplearning/tree/master/tf-obj).
