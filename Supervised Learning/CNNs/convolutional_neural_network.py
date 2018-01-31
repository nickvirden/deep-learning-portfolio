from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

###############
## BUILD CNN ##
###############

# Initialize CNN
classifier = Sequential()

# Step 1 - Convolution
# Step 2 - Pooling

# input_shape => reduces pics to 64x64 images with RGB
# Input Layer
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Hidden Layer 1
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection

# Complete first round of processing
classifier.add(Dense(units=128, activation='relu'))

# Output Layer
# activation is 'sigmoid' because we expect a binary outcome
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#######################
## FIT CNN TO IMAGES ##
#######################

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Increasing the image size from 64x64 to something bigger will get more information, also adding more layers can help as well
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# sudo pip install pillow
# Necessary for the classifier to process the images
from PIL import Image

classifier.fit_generator(
	training_set,
	steps_per_epoch=8000,
	epochs=25,
	validation_data=test_set,
	validation_steps=2000
	)

# Part 3 - Making New Predictions
import numpy as np
from keras.preprocessing import image

# Load one of our test images
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))

# Turn this image into a array of points
test_image = image.img_to_array(test_image)

# Turn the test image into a numpy array
test_image = np.expand_dims(test_image, axis=0)

# Get a prediction from our classifier
result = classifier.predict(test_image)

print "The training set has assigned the following indices to our cat and dog categories: " + str(training_set.class_indices)

# Get the prediction
prediction = ""

# If the result is equal to one, then the result is a dog
if result[0][0] == 1:
	prediction = 'dog'
# Otherwise, the prediction is a cat
else:
	prediction = 'cat'

print "The machine predicted: " + prediction

# Solutions with over 90% accuracy: https://www.udemy.com/deeplearning/learn/v4/questions/2276518