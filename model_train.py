import cv2
import os
import matplotlib.image as mpimg
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from tensorflow.keras.preprocessing import image

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
train_data = 'Train_processed'
test_data = 'Test_processed'

train_dir_normal = 'Train_processed/Normal'
train_dir_covid = 'Train_processed/Covid'
test_dir_normal = 'Test_processed/Normal'
test_dir_covid = 'Train_processed/Covid'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

training_data = train_datagen.flow_from_directory(directory=train_data,
                                                 target_size=(30,30),
                                                 batch_size=32,
                                                 class_mode='binary')
testing_data = test_datagen.flow_from_directory(directory=test_data,
                                               target_size = (30, 30),
                                               batch_size = 32,
                                               class_mode = 'binary')


print('Shape of Input image: {}'.format(training_data.image_shape))
print('Number of classes: {}'.format(len(set(training_data.classes))))


model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = training_data.image_shape))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.3))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.2))

model.add(Conv2D(filters = 128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(rate = 0.15))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(rate = 0.15))

model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(rate = 0.1))

model.add(Dense(units = 32, activation = 'relu'))
model.add(Dropout(rate = 0.1))

model.add(Dense(units = len(set(training_data.classes)), activation = 'softmax'))


print(model.summary())

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

fitted_model = model.fit(training_data,
                   steps_per_epoch=150,
                   epochs=50,
                   validation_data=testing_data,
                   validation_steps=150)


accuracy = fitted_model.history['accuracy']
plt.plot(range(len(accuracy)), accuracy, 'bo', label = 'accuracy',linestyle='dotted')
plt.legend()
plt.show()


test_image_path = 'Test_processed/Covid/Covid 246.jpeg'
test_image = np.expand_dims(image.img_to_array(image.load_img(test_image_path, target_size=(30,30))), axis=0)
result = model.predict(x=test_image)

if result[0][0] == 1:
    prediction = 'Covid'
else:
    prediction = 'Normal'
print(prediction)