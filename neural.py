import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

fashion_nist=keras.datasets.fashion_mnist  #load the dataset

(train_images, train_labels), (test_images, test_labels)= fashion_nist.load_data()
#print(train_images.shape)
#print(train_images[0,23,23])
#print(train_labels[:70])  #to look at the first ten training labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#plt.figure()
#plt.imshow(train_images[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

#Building the model
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  #input layer
    keras.layers.Dense(128, activation='relu'),  #hidden layer
    keras.layers.Dense(10, activation='softmax') #output layer
])
#This is telling keras how to train your model by specifying the loss, optimizer,metrics
model.compile(optmizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#train the model
model.fit(train_images, train_labels, epochs=10)
#To evaluate the model
test_loss, test_acc=model.evaluate(test_images, test_labels, verbose=1)
print('Test accurace: ', test_acc)
predictions=model.predict(test_images)


COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)
