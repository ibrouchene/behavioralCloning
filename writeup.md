# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[recovery]: ./Report/recovery.jpg "Recovery"
[recovery2]: ./Report/recovery2.jpg "Recovery2"
[bridgerecovery]: ./Report/bridgerecovery.jpg "bridgerecovery"

## Rubric Points
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes two sets of files.

One set corresponds to the initial model used, generated with relatively few images. 

The second set correspond to a model trained with much more data, in order to get the car further in the second track. 

Each of the sets contains the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* track1.mp4 (track2.mp4) containing the video of the autonomous drive
* The folder containing the training data

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The architecture chosen is based on the one from the NVIDIA paper "End to End Learning for Self-Driving Cars" [link](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

The model can be decomposed as such:
* Normalization layer
* Convolutional layers
* Fully connected layers

Before entering the normalization layer, the images are cropped in order to remove landscape information, and resized in order to help reduce processing time. Cropping is done using a built-in Keras layer, resizing and normalizing are done with a Lambda layer.

The model includes RELU layers to introduce nonlinearity (for instance in line 144).

#### 2. Attempts to reduce overfitting in the model

The model does not contains dropout layers. In order to reduce overfitting, the model was trained on all camera images, and each image was flipped in order to compensate for the left turn bias of track 1. 

The model was successfully tested on track1, and has proven to be good for keeping the car on the track for at least one lap.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 173).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

#### 3. Creation of the Training Set & Training Process

In order to get the model to work, I proceeded as such:

* Generate a very small set of training data in order to get the car all the way up to the first left turn

an initial try based on this dataset has proven to be already pretty effective. However I needed to "teach" the car to recover from moving too much on the side of the track. I therefore recorded a few recovery manoeuvers, as shown in this picture ![alt text][recovery]. 

* Iteratively generate recovery images for difficult portions

After that, the car was able to make it up to the bridge before crashing on the barrier, which led me to record a few more recovery images, on the bridge. ![alt text][bridgerecovery]. The last step was adding a few images for the left turn after the bridge (curve with sand on the side) and some other images to record a smooth drive over a curve.


That's it, with a surprisingly small amount of images (approx 5500), the car managed to learn good enough in order to complete one full lap around track 1 by itself!


As one could expect, letting the simulator run with this model on the second track shows the limits of the model. The car doesn't even make it to the first turn. In order to improve this, I proceeded the same way as I did for track 1: record more recovery data and a small amount of "normal" driving data. This has shown benefits for track2, but the performance on track1 was no longer satisfying the requirements. In order to improve the situation I recorded much more data for both track1 and track2, so that the network is forced to generalize. Along with "normal" drives I recorded some recovery moves.

This extension of the training set has proven to be beneficial to the model, the car was able to make an entire lap on track1 and get much further away on track2. This can be seen in the corresponding video files under the folder "Intermediate_model"


Before being fed in the network, the training data was shuffled and 20% of the available data was set aside for validation. After a few trials a number of epoch of 3 has proven to be enough for the task at hand. As mentionned before, there was no need to tune the learning rate since the network was being trained with an adam optimizer.

