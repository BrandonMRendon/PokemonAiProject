from tensorflow import keras
from tensorflow.keras import layers
from extra_keras_datasets import emnist
import matplotlib.pyplot as plt
import numpy as np
import cv2, pickle, PIL


def get_text(image):  # ML model that gets the text on screen
    model = keras.models.load_model('EMNIST.model')
    # Convert image
    model_string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    result_string = ""
    for i in range(len(image)):
        image_data = np.asarray(image)
        result = np.argmax(model.predict(image_data), axis=-1)
        result_string += model_string[result[i]]
    print(result_string)


def make_model():
    # Load dataset from emnist https://github.com/christianversloot/extra_keras_datasets#emnist-balanced
    (x_train, y_train), (x_test, y_test) = emnist.load_data(type='byclass')
    x_train[0].shape
    (28, 28)
    x_train, x_test = x_train / 255, x_test / 255  # greyscale

    # Model structure
    model = keras.Sequential(name='EMNIST_Model')
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(62, activation='softmax'))

    # Compile Model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fitting Model
    model.fit(x_train, y_train, epochs=5)

    # Evaluate
    model.evaluate(x_test, y_test)


    model.save('EMNIST.model')

    # Predict
    # plt.imshow(x_test[100])
    # plt.imshow(image)

    # print(image_data.shape)
    # print(x_train.shape)

    # Save
    # model.save('MNIST_DATA.model')


def read_text():
    image = cv2.imread('Venusaur_Set.png')  # BGR image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    (height, width, depth) = image.shape
    '''
    print('height={}, width={}, depth={}'.format(height, width, depth))
    plt.imshow(image)
    plt.show()
    '''
    move_area = image[32:141, 194:587]
    # plt.imshow(move_area)

    # Get the coordinates of 4 moves
    move_1 = image[39:58, 195:375]
    move_2 = image[79:98, 195:344]
    move_3 = image[65:84, 435:586]
    move_4 = image[105:124, 405:586]
    # plt.imshow(move_4)

    # Get coordinates of each letter, split into separate images

    gray = cv2.cvtColor(move_1, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('Image', thresh)
    arr = np.asarray(thresh)
    # print(arr[2])
    '''
    All letters in screenshots are 11x15 (11 wide, 15 tall)
    This means we can transverse the array from left to right going down every single time until this algorithm
    hits its first black pixel. We then calculate the size of this letter, save the coordinates, and skip
    over the rest of the letter.
    While loop is used here to increment 11 instead of 1.
    '''

    letters_x = []
    letters_y = []
    x_size = len(arr[0])
    i = 0
    while i < x_size:
        bp = False  # black pixel or 0
        for j in range(len(arr)):
            if arr[j][i] == 0:
                bp = True
        if bp:
            letters_x.append([i, i+11])  # letter is 11 wide, get start and end x coordinates
            letters_y.append([0, len(arr)-1])  # letter is 15 tall, but get height of the whole image
            i += 12  # Increment past this letter to the next spot
        else:
            i += 1
    # print(letters_x)
    # print(letters_y)
    letter_1 = gray[letters_y[0][0]:letters_y[0][1], letters_x[0][0]:letters_x[0][1]]
    letter_1 = cv2.threshold(letter_1, 127, 255, cv2.THRESH_BINARY)[1]
    letter_1 = cv2.resize(letter_1, (28, 28))
    letter_1 = cv2.copyMakeBorder(letter_1, 5, 5, 8, 9, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.imshow('Image', letter_1)

    letters = []
    for i in range(len(letters_x)):
        letter = (gray[letters_y[i][0]:letters_y[i][1], letters_x[i][0]:letters_x[i][1]])
        letter = cv2.copyMakeBorder(letter, 5, 5, 8, 9, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        letters.append(letter)
    get_text(letters)

