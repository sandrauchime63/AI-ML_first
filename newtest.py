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

#the images are different sizes so we need to resize (preprocessing)
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

BATCH_SIZE=32
SHUFFLE_BUFFER_SIZE=1000

train_batches=train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches=validationn.batch(BATCH_SIZE)
test_batches=test.batch(BATCH_SIZE)
'''
for image_batch, label_batch in train.take(1):
    print(image_batch.shape)
'''
#Using a pre-trained model
IMG_SHAPE=(IMG_SIZE, IMG_SIZE, 3)
base_model=tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
base_model.trainable=False #freeze the convolutional base
base_model.summary()
#Adding our own classifier
global_average_layer=tf.keras.layers.GlobalAveragePooling2D()
prediction_layer=tf.keras.layers.Dense(1) #binary classification (cats vs dogs)
model=tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.summary()
#compile the model
base_learning_rate=0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
#Train the model
initial_epochs=3
validation_steps=20
loss0, accuracy0=model.evaluate(validation_batches, steps=validation_steps)


history= model.fit(train_batches,
            epochs=initial_epochs,
            validation_data=validation_batches)
history_dict=history.history['accuracy']
#Evaluate the model
loss1, accuracy1=model.evaluate(test_batches)
print("Test accuracy after initial training: {:.2f}".format(accuracy1))
model.save('cats_vs_dogs.h5')
