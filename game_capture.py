from tensorflow import keras
from tensorflow.keras import layers
from extra_keras_datasets import emnist
import matplotlib.pyplot as plt
import numpy as np
import cv2, pickle, PIL


move_list = ["ABSORB", "ACID", "ACIDARMOR", "AGILITY", "AMNESIA", "AURORABEAM", "BARRAGE", "BARRIER", "BIDE", "BIND", "BITE", "BLIZZARD", "BODYSLAM", "BONECLUB", "BONEMERANG", "BUBBLE", "BUBBLEBEAM", "CLAMP", "COMETPUNCH", "CONFUSERAY", "CONFUSION", "CONSTRICT", "CONVERSION", "COUNTER", "CRABHAMMER", "CUT", "DEFENSECURL", "DIG", "DISABLE", "DIZZYPUNCH", "DOUBLEKICK", "DOUBLESLAP", "DOUBLETEAM", "DOUBLE-EDGE", "DRAGONRAGE", "DREAMEATER", "DRILLPECK", "EARTHQUAKE", "EGGBOMB", "EMBER", "EXPLOSION", "FIREBLAST", "FIREPUNCH", "FIRESPIN", "FISSURE", "FLAMETHROWER", "FLASH", "FLY", "FOCUSENERGY", "FURYATTACK", "FURYSWIPES", "GLARE", "GROWL", "GROWTH", "GUILLOTINE", "GUST", "HARDEN", "HAZE", "HEADBUTT", "HIGHJUMPKICK", "HORNATTACK", "HORNDRILL", "HYDROPUMP", "HYPERBEAM", "HYPERFANG", "HYPNOSIS", "ICEBEAM", "ICEPUNCH", "JUMPKICK", "KARATECHOP", "KINESIS", "LEECHLIFE", "LEECHSEED", "LEER", "LICK", "LIGHTSCREEN", "LOVELYKISS", "LOWKICK", "MEDITATE", "MEGADRAIN", "MEGAKICK", "MEGAPUNCH", "METRONOME", "MIMIC", "MINIMIZE", "MIRRORMOVE", "MIST", "NIGHTSHADE", "PAYDAY", "PECK", "PETALDANCE", "PINMISSILE", "POISONGAS", "POISONPOWDER", "POISONSTING", "POUND", "PSYBEAM", "PSYCHIC", "PSYWAVE", "QUICKATTACK", "RAGE", "RAZORLEAF", "RAZORWIND", "RECOVER", "REFLECT", "REST", "ROAR", "ROCKSLIDE", "ROCKTHROW", "ROLLINGKICK", "SANDATTACK", "SCRATCH", "SCREECH", "SEISMICTOSS", "SELF-DESTRUCT", "SHARPEN", "SING", "SKULLBASH", "SKYATTACK", "SLAM", "SLASH", "SLEEPPOWDER", "SLUDGE", "SMOG", "SMOKESCREEN", "SOFT-BOILED", "SOLARBEAM", "SONICBOOM", "SPIKECANNON", "SPLASH", "SPORE", "STOMP", "STRENGTH", "STRINGSHOT", "STRUGGLE", "STUNSPORE", "SUBMISSION", "SUBSTITUTE", "SUPERFANG", "SUPERSONIC", "SURF", "SWIFT", "SWORDSDANCE", "TACKLE", "TAILWHIP", "TAKEDOWN", "TELEPORT", "THRASH", "THUNDER", "THUNDERPUNCH", "THUNDERSHOCK", "THUNDERWAVE", "THUNDERBOLT", "TOXIC", "TRANSFORM", "TRIATTACK", "TWINEEDLE", "VINEWHIP", "VISEGRIP", "WATERGUN", "WATERFALL", "WHIRLWIND", "WINGATTACK", "WITHDRAW", "WRAP"]


def get_text(image):  # ML model that gets the text on screen
    model = keras.models.load_model('EMNIST.model')
    # Convert image
    model_string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    result_string = ""
    for i in range(len(image)):
        image_data = np.asarray(image)
        result = np.argmax(model.predict(image_data), axis=-1)
        result_string += model_string[result[i]]
    # print("I see \"" + result_string + "\"")
    # print("I predict this is \"" + correct_move_string(result_string) + "\"")
    return correct_move_string(result_string)


def correct_move_string(s):
    result = ""
    wrong = 0
    prediction_index = -1
    lowest = 13  # I think the largest possible move has 13 characters
    for i in range(len(move_list)):
        if len(s) == len(move_list[i]):  # See if the current move string is the same length as one in the move list
            for j in range(len(s)):
                if move_list[i][j] != s[j]:
                    wrong += 1
            if wrong < lowest:
                lowest = wrong
                prediction_index = i
            # print("Name: " + move_list[i] + " Wrong: " + str(wrong))
        wrong = 0
    result = move_list[prediction_index]
    return result


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
        optimizer='nadam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Fitting Model
    model.fit(x_train, y_train, epochs=10)

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


def get_move_list():
    # Load an image into RGB colorspace from disk
    image = cv2.imread('Venusaur_Set.png')  # BGR image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the coordinates of 4 moves
    move_1 = image[90:130, 682:1088]
    move_2 = image[180:220, 682:1022]
    move_3 = image[148:188, 1225:1565]
    move_4 = image[238:278, 1157:1566]
    # plt.imshow(move_4)
    moves = [move_1, move_2, move_3, move_4]
    '''
    print(get_letters_from_move(move_1))
    print(get_letters_from_move(move_2))
    print(get_letters_from_move(move_3))
    print(get_letters_from_move(move_4))
    '''

    # Process image in opencv and save
    image_draw = image.copy()
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
    cv2.rectangle(image_draw, (680, 88), (1090, 132), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_move(move_1), (675, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(image_draw, (680, 178), (1024, 222), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_move(move_2), (675, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(image_draw, (1223, 146), (1563, 190), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_move(move_3), (1210, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(image_draw, (1155, 236), (1564, 280), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_move(move_4), (1155, 336), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imwrite('result.png', image_draw)


def get_letters_from_move(move):

    '''
    All letters in screenshots (except 'I') are 26x38 (26 wide, 38 tall)
    This means we can transverse the array from left to right going down every single time until this algorithm
    hits its first black pixel. We then calculate the size of this letter, save the coordinates, and skip
    over the rest of the letter.
    While loop is used here to increment 27 instead of 1.
    '''

    gray = cv2.cvtColor(move, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    arr = np.asarray(thresh)

    letters_x = []
    letters_y = []
    x_size = len(arr[0])
    j = 0
    i_check = False
    i_found = False
    while j < x_size:
        bp = False  # black pixel or 0
        for k in range(len(arr)):
            if arr[k][j] == 0:
                bp = True
        if bp:
            if i_check:  # pixels 5 and 35 black and previous is blank, letter = I
                bp_count = 0
                for l in range(len(arr)):
                    if arr[l][j] == 0:
                        bp_count += 1
                if (bp_count == 2) and (arr[5][j] == 0) and (arr[35][j] == 0):
                    # print("found waldo!")
                    i_found = True
            if not i_found:
                letters_x.append([j, j+26])  # letter is 26 wide, get start and end x coordinates
                letters_y.append([0, len(arr)-1])  # get height of the letter from word cutout
                j += 27  # Increment past this letter to the next spot
            else:
                letters_x.append([j-6, j+20]) # get 6 pixels before I to 20 pixels after this index
                letters_y.append([0, len(arr)-1]) # get height of the letter from word cutout
                j += 21
                i_found = False
            i_check = False
        else:
            i_check = True
            j += 1
    # print(letters_x)

    # Get each individual letter in Thresh_Binary (NOT inverted!) using coordinates
    letters = []
    for j in range(len(letters_x)):
        letter = (gray[letters_y[j][0]:letters_y[j][1], letters_x[j][0]:letters_x[j][1]])
        letter = cv2.threshold(letter, 180, 255, cv2.THRESH_BINARY)[1]
        letter = cv2.resize(letter, (20, 20))
        letter = cv2.copyMakeBorder(letter, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        letters.append(letter)
    return get_text(letters)

