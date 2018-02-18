import os
import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout
from keras import backend as K

STEERING_COMPENSATION = 0.08  # Set experimentally in order to reproduce the same trajectory than with the center camera on a straigth section
TRAINING_SET = "Step2"

# Read the csv file
samples = []
with open('./Step2/' + "driving_log.csv") as data:
    reader = csv.reader(data)
    for line in reader:
        samples.append(line)

# Training and validation split        
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

      
def generator(samples, batch_size=32):
    """ Generator for providing input data """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # CENTER IMAGE
                imname = './Step2/IMG/' + batch_sample[0].split('\\')[-1]
                image = cv2.imread(imname)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3])
                image_flipped = np.fliplr(image) # Flip image in order to remove the left turn bias
                angle_flipped = -angle # Invert sign of steering wheel angle in order to match the flipped image
                # make sure to append items in the correct order!
                images.append(image)
                angles.append(angle)
                images.append(image_flipped)
                angles.append(angle_flipped)
                # LEFT
                imname = './Step2/IMG/' + batch_sample[1].split('\\')[-1]
                image = cv2.imread(imname)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3]) + STEERING_COMPENSATION
                image_flipped = np.fliplr(image) # Flip image in order to remove the left turn bias
                angle_flipped = -angle # Invert sign of steering wheel angle in order to match the flipped image
                # make sure to append items in the correct order!
                images.append(image)
                angles.append(angle)
                images.append(image_flipped)
                angles.append(angle_flipped)
                # RIGHT
                imname = './Step2/IMG/' + batch_sample[2].split('\\')[-1]
                image = cv2.imread(imname)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                angle = float(batch_sample[3]) - STEERING_COMPENSATION
                image_flipped = np.fliplr(image) # Flip image in order to remove the left turn bias
                angle_flipped = -angle # Invert sign of steering wheel angle in order to match the flipped image
                # make sure to append items in the correct order!
                images.append(image)
                angles.append(angle)
                images.append(image_flipped)
                angles.append(angle_flipped)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train,y_train)


def LoadTrainingData():
    lines = []
    with open(os.path.join(TRAINING_SET, "driving_log.csv")) as data:
        reader = csv.reader(data)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    for line in lines[1:]:
        # process center image
        center_image = line[0]
        filename = center_image.split('/')[-1]
        current_path = os.path.join(TRAINING_SET, "IMG", filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)
        
        
        # process left image
        left_image = line[1]
        filename = left_image.split('/')[-1]
        current_path = os.path.join(TRAINING_SET, "IMG", filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3]) + STEERING_COMPENSATION
        measurements.append(measurement)
        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)
        
        #process right image
        right_image = line[2]
        filename = right_image.split('/')[-1]
        current_path = os.path.join(TRAINING_SET, "IMG", filename)
        image = cv2.imread(current_path)
        images.append(image)
        measurement = float(line[3]) - STEERING_COMPENSATION
        measurements.append(measurement)
        image_flipped = np.fliplr(image)
        measurement_flipped = -measurement
        images.append(image_flipped)
        measurements.append(measurement_flipped)
        
    return np.array(images), np.array(measurements)

if __name__ == '__main__':
    
    train_generator      = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # CNN network very similar to the one from NVIDIA
    model = Sequential()
    
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3))) # Crop the parts of the images that are not interesting
    
    model.add(Lambda(lambda x: K.tf.image.resize_images(x, (66, 200))))  # Resize the image in order to win a bit of processing time
    
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))  # Normalize input data
    
    # First convolutional layers: 5x5 kernel with 2 pixels strides
    model.add(Conv2D(3, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
    
    model.add(Conv2D(24, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
    
    model.add(Conv2D(36, 5, 5, subsample=(2,2), border_mode='same', activation='relu'))
    
    
    # Second group of convolutional layers: 3x3 kernel with 1 pixel strides
    
    model.add(Conv2D(48, 3, 3, subsample=(1,1), border_mode='same', activation='relu'))
    
    model.add(Conv2D(64, 3, 3, subsample=(1,1), border_mode='same', activation='relu'))
    
    # Flatten and wrap up with fully connected layers
    
    model.add(Flatten())
    
    model.add(Dense(100, activation='relu'))
    
    model.add(Dense(50, activation='relu'))
    
    model.add(Dense(10, activation='relu'))
    
    model.add(Dense(1))
    
    
    # print network architecture
    model.summary()
    
    model.compile(loss='mse', optimizer='adam')
    
    model.fit_generator(train_generator,
                        samples_per_epoch= (len(train_samples)*6.0),
                        validation_data=validation_generator, 
                        nb_val_samples=len(validation_samples), 
                        nb_epoch=5)
    
    model.save('model_Step2.h5')
    print("model saved")
    K.clear_session()