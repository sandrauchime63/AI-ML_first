
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.keras.datasets import mnist

data_keras=tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test)=data_keras.load_data() #x_train holds the images and y_train holds the digit each image represents
#print(x_train.shape)
print(x_train[0, 10, 15])
#print(y_train[:10])

class_number=["S", "O", "U", "I", "A", "2", "1", "3", "i", "4"]

#To show the first 10 images in the dataset
'''
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.title(class_number[y_train[i]])
    plt.axis("off")
    plt.show()

'''
#make training easier by dividing by 255 and turning pixel to 0 and 1
x_train=x_train/255.0
x_test=x_test/255.0

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(123, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

test_loss, test_acc=model.evaluate(x_test, y_test, verbose=1)
#print("Test accuracy: ", test_acc)

prediction=model.predict(x_train)
#print(prediction[0])

def predict(model, image, correct_label):
    class_number=["S", "O", "U", "I", "A", "2", "1", "3", "i", "4"]
    predict=model.predict(np.array([image]))
    predicted_class=class_number[np.argmax(predict)]

    def show_image(img, label, guess):
        plt.figure()
        plt.imshow(img, cmap=plt.cm.binary)
        plt.title("Expected: " + label)
        plt.xlabel("Guess: " + guess)
        plt.colorbar()
        plt.grid(False)
        plt.show()

    show_image(image, class_number[correct_label], predicted_class)

def get_numbers():
    while True:
        num=input("Please put in a number: ")
        if num.isdigit():
            num=int(num)
            if 0<=num<=1000:
                return int(num)
        else:
            print("try again...")

num= get_numbers()
image=x_test(num)
label=y_test(num)
predict(model, image, label)
