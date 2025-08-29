#The point of this is to be able to tell which image belongs to what class
#we have 10(0-9) classes to fit every image into

import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import datasets, layers, models
#from keras.models import load_model, Sequential
from keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.ke

data_keras=keras.datasets.mnist
(x_train, y_train), (x_test, y_test)=data_keras.load_data()
#print(x_train.shape)
#print(x_test[:100])
#There are ten classes
#plt.figure()
#plt.imshow(x_train[10])
#plt.show()

classes=["S", "O", "U", "I", "A", "2", "1", "3", "i", "4"]
x_train=x_train/255.0
x_test=x_test/255.0


model=models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history=model.fit(x_train, y_train, epochs=10,
                  validation_data=(x_test, y_test))
test_loss, test_acc=model.evaluate(x_test, y_test, verbose=2)
print(test_acc)


#Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
datagen=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_img=x_train[0] 
img=image.img_to_array(test_img)#convert to Numpy array
img=img.reshape((1,) + img.shape) #this is now a Numpy array

i=0
for batch in datagen.flow(img, batch_size=1):
    plt.figure(i)
    imgplot=plt.imshow(image.array_to_img(batch[0]))
    i+=1
    if i%4==0:
        break
plt.show()