import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from extra_keras_datasets import emnist
import matplotlib.pyplot as plt
import numpy as np
import cv2, pickle, PIL, math

# Temporary, remove later in favor of calling Pokedex class accessor methods.
move_list = ["ABSORB", "ACID", "ACIDARMOR", "AGILITY", "AMNESIA", "AURORABEAM", "BARRAGE", "BARRIER", "BIDE", "BIND", "BITE", "BLIZZARD", "BODYSLAM", "BONECLUB", "BONEMERANG", "BUBBLE", "BUBBLEBEAM", "CLAMP", "COMETPUNCH", "CONFUSERAY", "CONFUSION", "CONSTRICT", "CONVERSION", "COUNTER", "CRABHAMMER", "CUT", "DEFENSECURL", "DIG", "DISABLE", "DIZZYPUNCH", "DOUBLEKICK", "DOUBLESLAP", "DOUBLETEAM", "DOUBLE-EDGE", "DRAGONRAGE", "DREAMEATER", "DRILLPECK", "EARTHQUAKE", "EGGBOMB", "EMBER", "EXPLOSION", "FIREBLAST", "FIREPUNCH", "FIRESPIN", "FISSURE", "FLAMETHROWER", "FLASH", "FLY", "FOCUSENERGY", "FURYATTACK", "FURYSWIPES", "GLARE", "GROWL", "GROWTH", "GUILLOTINE", "GUST", "HARDEN", "HAZE", "HEADBUTT", "HIGHJUMPKICK", "HORNATTACK", "HORNDRILL", "HYDROPUMP", "HYPERBEAM", "HYPERFANG", "HYPNOSIS", "ICEBEAM", "ICEPUNCH", "JUMPKICK", "KARATECHOP", "KINESIS", "LEECHLIFE", "LEECHSEED", "LEER", "LICK", "LIGHTSCREEN", "LOVELYKISS", "LOWKICK", "MEDITATE", "MEGADRAIN", "MEGAKICK", "MEGAPUNCH", "METRONOME", "MIMIC", "MINIMIZE", "MIRRORMOVE", "MIST", "NIGHTSHADE", "PAYDAY", "PECK", "PETALDANCE", "PINMISSILE", "POISONGAS", "POISONPOWDER", "POISONSTING", "POUND", "PSYBEAM", "PSYCHIC", "PSYWAVE", "QUICKATTACK", "RAGE", "RAZORLEAF", "RAZORWIND", "RECOVER", "REFLECT", "REST", "ROAR", "ROCKSLIDE", "ROCKTHROW", "ROLLINGKICK", "SANDATTACK", "SCRATCH", "SCREECH", "SEISMICTOSS", "SELF-DESTRUCT", "SHARPEN", "SING", "SKULLBASH", "SKYATTACK", "SLAM", "SLASH", "SLEEPPOWDER", "SLUDGE", "SMOG", "SMOKESCREEN", "SOFT-BOILED", "SOLARBEAM", "SONICBOOM", "SPIKECANNON", "SPLASH", "SPORE", "STOMP", "STRENGTH", "STRINGSHOT", "STRUGGLE", "STUNSPORE", "SUBMISSION", "SUBSTITUTE", "SUPERFANG", "SUPERSONIC", "SURF", "SWIFT", "SWORDSDANCE", "TACKLE", "TAILWHIP", "TAKEDOWN", "TELEPORT", "THRASH", "THUNDER", "THUNDERPUNCH", "THUNDERSHOCK", "THUNDERWAVE", "THUNDERBOLT", "TOXIC", "TRANSFORM", "TRIATTACK", "TWINEEDLE", "VINEWHIP", "VISEGRIP", "WATERGUN", "WATERFALL", "WHIRLWIND", "WINGATTACK", "WITHDRAW", "WRAP"]
pokemon_list = ["VENUSAUR", "CHARIZARD", "BLASTOISE"]


def get_text(image, is_pokemon):  # ML model that gets the text on screen
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
    if is_pokemon:
        return correct_pokemon_string(result_string)
    else:
        return correct_move_string(result_string)


def get_numbers(image):  # ML model that gets the numbers on screen
    model = keras.models.load_model('MNIST.model')
    model_string = "0123456789"
    result_string = ""
    for i in range(len(image)):
        image_data = np.asarray(image)
        result = np.argmax(model.predict(image_data), axis=-1)
        result_string += model_string[result[i]]
        # if result[i] == 4:
            # cv2.imshow('Image', image[i])
    # print("I see \"" + result_string + "\"")
    return result_string


def correct_pokemon_string(s):
    result = ""
    wrong = 0
    prediction_index = -1
    lowest = 13  # I think the largest possible pokemon has 13 characters
    for i in range(len(pokemon_list)):
        if len(s) == len(pokemon_list[i]):  # See if the current pokemon string is the same length as one in the move list
            for j in range(len(s)):
                if pokemon_list[i][j] != s[j]:
                    wrong += 1
            if wrong < lowest:
                lowest = wrong
                prediction_index = i
            # print("Name: " + pokemon_list[i] + " Wrong: " + str(wrong))
        wrong = 0
    result = pokemon_list[prediction_index]
    return result


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


def make_model_mnist():
    # Make a mnist model from mnist dataset in keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train[0].shape
    (28, 28)
    x_train, x_test = x_train / 255, x_test / 255  # greyscale

    # Create model structure
    model = keras.Sequential(name='MNIST_Model')
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile Model
    model.compile(
        optimizer='nadam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_train, y_train, epochs=5)
    model.save('MNIST.model')


def make_model_emnist():
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


def get_everything(filename):
    get_pokemon_names(filename)
    get_move_list('result.png')
    get_hp('result.png')


def get_pokemon_names(f):
    # Load an image into RGB colorspace from disk
    image = cv2.imread(f)  # RGB image
    ally = image[156:207, 345:652]
    enemy = image[751:809, 1272:1570]
    # cv2.imshow('Image', ally)
    # print(get_letters_from_word(ally, True))

    image_draw = image.copy()
    cv2.rectangle(image_draw, (343, 154), (654, 209), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_word(ally, True), (345, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(image_draw, (1270, 749), (1572, 811), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_word(enemy, True), (1250, 740), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imwrite('result.png', image_draw)


def get_hp_from_bar(max_health, f, ally):
    """Attempts to predict current hp value from a screenshot by looking at the HP bar in an image.
    Thresholds the image based on what color the HP bar is.

    arguments:
    max_health - The pokemon's max health, as an integer.
    f - File path to image as a string. Should be 'Something.png' or 'Something.jpg'.
    ally - A boolean, True if the health bar being inquired is the ally pokemon, False if enemy.
    """
    image = cv2.imread(f)  # RGB image
    if ally:
        image = image[262:277, 421:635]
    else:
        image = image[864:879, 1343:1557]
    # print("RGB values: " + str(image[7][0][2]) + " " + str(image[7][0][1]) + " " + str(image[7][0][0]))

    # Go pixel 7 down in the first column and check its RGB value. Then threshold the image based on what color it is.
    if image[7][0][0] > image[7][0][2]:  # Green
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]  # Case Green, tested extensively, 100 in binary works well
    elif image[7][0][1] < image[7][0][0]:  # Red
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.threshold(image, 52, 255, cv2.THRESH_BINARY_INV)[1]  # Case Red, tested, 52 in binary INVERSION(!) works well
    else:  # Yellow
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]  # Case Yellow, needs more testing but 100 in binary seems to be fine
    # cv2.imshow('Image', image)

    # Loop to get number of pixels in thresholded image that are white and return result
    arr = np.asarray(image)
    total_pixels = 210
    current_pixels = 0
    center = 5
    full = False
    i = 4     # start 4 pixels in instead of 0, hp bar representing "1 hp" is not just 1 pixel
    while i < total_pixels:
        if arr[center][i] == 255:
            current_pixels += 1
        i += 1
    if arr[center][210] == 255: current_pixels += 1
    if arr[center][211] == 255: current_pixels += 1
    if arr[center][212] == 255: current_pixels += 1
    if arr[center][213] == 255: current_pixels += 1
    if not full:
        current_health = math.ceil(max_health * (current_pixels/total_pixels))
    else:
        current_health = math.ceil(max_health * (current_pixels / total_pixels)) + 3

    print("I see " + str(current_pixels) + " pixels out of 214.")
    print("I predict this is " + str(current_health) + "/" + str(max_health))
    return current_health


def get_hp(f):
    image = cv2.imread(f)  # RGB image
    ally_hp_current = image[280:317, 370:501]
    ally_hp_max = image[280:317, 532:652]
    enemy_hp_current = image[882:917, 1297:1422]
    enemy_hp_max = image[882:917, 1454:1572]

    # Draw rectangles
    image_draw = image.copy()
    cv2.rectangle(image_draw, (368, 278), (503, 319), (0, 0, 255), 2)
    cv2.rectangle(image_draw, (530, 278), (654, 319), (0, 0, 255), 2)
    # cv2.putText(image_draw, get_digits_from_number(ally_hp_current) + "/", (520, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(image_draw, get_digits_from_number(ally_hp_current) + "/" + get_digits_from_number(ally_hp_max), (520, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(image_draw, (1295, 880), (1424, 919), (0, 0, 255), 2)
    cv2.rectangle(image_draw, (1452, 880), (1574, 919), (0, 0, 255), 2)
    cv2.putText(image_draw, get_digits_from_number(enemy_hp_current) + "/" + get_digits_from_number(enemy_hp_max), (1110, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.imshow('Image', image_draw)
    cv2.imwrite('result.png', image_draw)


def get_move_list(f):
    # Load an image into RGB colorspace from disk
    image = cv2.imread(f)  # RGB image

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
    cv2.rectangle(image_draw, (680, 88), (1090, 132), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_word(move_1, False), (675, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(image_draw, (680, 178), (1024, 222), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_word(move_2, False), (675, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(image_draw, (1223, 146), (1563, 190), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_word(move_3, False), (1210, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.rectangle(image_draw, (1155, 236), (1564, 280), (0, 255, 0), 2)
    cv2.putText(image_draw, get_letters_from_word(move_4, False), (1155, 336), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imwrite('result.png', image_draw)


def get_letters_from_word(word, is_pokemon):

    """
    All letters in screenshots (except 'I') are 26x38 (26 wide, 38 tall)
    This means we can transverse the array from left to right going down every single time until this algorithm
    hits its first black pixel. We then calculate the size of this letter, save the coordinates, and skip
    over the rest of the letter.
    While loop is used here to increment 27 instead of 1.
    """

    gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    # cv2.imshow('Image', thresh)
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
    return get_text(letters, is_pokemon)


def get_digits_from_number(number):
    gray = cv2.cvtColor(number, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow('Image', thresh)
    arr = np.asarray(thresh)

    numbers_x = []
    numbers_y = []
    x_size = len(arr[0])
    j = 0
    i_check = False
    i_found_case_1 = False
    i_found_case_2 = False
    i_found_case_3 = False
    while j < x_size:
        bp = False  # black pixel or 0
        for k in range(len(arr)):
            if arr[k][j] == 0:
                bp = True
        if bp:
            if i_check:
                bp_count = 0
                for l in range(len(arr)):
                    if arr[l][j] == 0:
                        bp_count += 1
                if (bp_count == 1) & (arr[8][j] == 0):
                    print("found waldo!")
                    i_found_case_1 = True
                if (bp_count == 5) & (arr[7][j] == 0) & (arr[8][j] == 0) & (arr[9][j] == 0) & (arr[28][j] == 0) & (arr[29][j] == 0):
                    print("found waldo 2!")
                    i_found_case_2 = True
                if (bp_count == 3) & (arr[7][j] == 0) & (arr[8][j] == 0) & (arr[29][j] == 0):
                    print("found waldo 3!")
                    i_found_case_3 = True
            if not (i_found_case_1 | i_found_case_2 | i_found_case_3):
                numbers_x.append([j, j + 32])
                numbers_y.append([0, len(arr) - 1])
                j += 33
            elif i_found_case_3:
                numbers_x.append([j - 13, j + 18])
                numbers_y.append([0, len(arr) - 1])
                j += 20
                i_found_case_3 = False
            elif i_found_case_2:
                numbers_x.append([j - 13, j + 18])
                numbers_y.append([0, len(arr) - 1])
                j += 20
                i_found_case_2 = False
            elif i_found_case_1:
                numbers_x.append([j - 10, j + 21])
                numbers_y.append([0, len(arr) - 1])
                j += 22
                i_found_case_1 = False
            i_check = False
        else:
            i_check = True
            j += 1
    # print(letters_x)

    # Get each individual letter in Thresh_Binary (NOT inverted!) using coordinates
    numbers = []
    for j in range(len(numbers_x)):
        number = (gray[numbers_y[j][0]:numbers_y[j][1], numbers_x[j][0]:numbers_x[j][1]])
        number = cv2.threshold(number, 166, 255, cv2.THRESH_BINARY)[1]
        number = cv2.resize(number, (12, 22))
        number = cv2.copyMakeBorder(number, 3, 3, 8, 8, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        numbers.append(number)
    # plt.imshow(numbers[1])
    return get_numbers(numbers)

