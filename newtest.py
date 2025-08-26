import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#loading the dataset
(raw_train, raw_validation, raw_test), metadata=tfds.load(
'cats_vs_dogs',
split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
with_info=True,
as_supervised=True,
)
get_label=metadata.features['label'].int2str #this is used to turn class labels(0,1) back to their representing string names

#To show 2 images from the dataset
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label(label))

#the images are different sizes so we need to resize
IMG_SIZE=160
def format_example(image, label):
    #returns an image that is resized
    image=tf.cast(image, tf.float32) #Changes it to a tensorflow tnsor and floating point number
    image=tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) #resize it 
    image=image/255 #divide it by half of 255 to put it back to 0 and 1
    return (image, label)

#apply the resizing to all images
train=raw_train.map(format_example)
validationn=raw_validation.map(format_example)
test=raw_test.map(format_example)








