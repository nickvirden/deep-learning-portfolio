# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

#################################
### PART 1 - BUILDING THE CNN ###
#################################

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# Add a convolution layer
# nb_filter => number of feature detectors / filters / convolution kernel
# nb_rows => row dimension of filter
# nb_columns => column dimension of filter
# input_shape => specify input format of images
#             => default is 3 channels, 256 rows, 256 columns (i.e. color 256x256 picture)
#             => NOTE: images are converted into 2D arrays if B&W, 3D arrays if COLOR
#             => In TENSORFLOW, input_shape=(row, column, channels)
#             => In THEANO, input_shape=(channels, rows, columns)
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
# pool_size => size of filter that's applied during max pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
# We don't lose a lot of data when we map features because we extracted only the most important, identifiable data about the picture when we filtered them
# The numbers in the Pooled image represent specific features, not the more general, useless information of a single pixel

# Keras understands that we need to flatten the previous layer from the previous step
classifier.add(Flatten())

# Step 4 - Full Connection
# output_dim => a number around 100 is about right based on our dimensions after convolution and max pooling, but it is common to pick a power of 2 as your output dim

# Fully connected layer
classifier.add(Dense(output_dim=128, activation='relu'))

# Output layer
classifier.add(Dense(output_dim=2, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

##########################################
### PART 2 - FITTING THE CNN TO IMAGES ###
##########################################
# Keras (https://keras.io/preprocessing/image/) allows for image augmentation, which means we can have a small number of images, but produce a varied training set on which to train our CNN

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
	rescale=1./255, # rescales all pixel values to between 0 and 1
	shear_range=0.2, # random cropping values
	zoom_range=0.2, # random zoom value
	horizontal_flip=True # random horizontal flips
)

# Resizes images
test_datagen = ImageDataGenerator(rescale=1./255)

# Creates training set
# Something important to note here is that if you separate the data into folders, Keras will automatically count up all the images AND detect that the folders correspond to separate classes
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Creates test set
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Fits CNN to to training set and tests its performance against the test set
classifier.fit_generator(
    training_set,
    steps_per_epoch=8000, # number of images we have in our training set
    epochs=25, # number of iterations
    validation_data=test_set, # how we validate our performance
    validation_steps=2000 
)