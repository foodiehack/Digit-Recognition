import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
import requests
from PIL import Image
import cv2
import random
np.random.seed(0)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X_test, y_test,verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
def recognition(url):
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    img = np.asarray(img)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    bgcolor = input("Is background color black or white or not applicable(B/W/N)")
    if(bgcolor == 'B' or bgcolor == 'b'):
            img = img/255
            img2 = img.reshape(1,28,28,1)
            prediction = loaded_model.predict_classes(img2)
            print("predicted digit:", str(prediction))
    elif(bgcolor == 'W' or bgcolor == 'w'):
            img = cv2.bitwise_not(img)
            img = img/255
            img2 = img.reshape(1,28,28,1)
            prediction = loaded_model.predict_classes(img2)
            print("predicted digit:", str(prediction))
    else:
            print("Provide picture according to the specified parameters")

def recognition_pc(path):
    img = Image.open(path)
    img = np.asarray(img)
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.show()
    bgcolor = input("Is background color black or white or not applicable(B/W/N)")
    if(bgcolor == 'B' or bgcolor == 'b'):
            img = img/255
            img2 = img.reshape(1,28,28,1)
            prediction = loaded_model.predict_classes(img2)
            print("predicted digit:", str(prediction))
    elif(bgcolor == 'W' or bgcolor == 'w'):
            img = cv2.bitwise_not(img)
            img = img/255
            img2 = img.reshape(1,28,28,1)
            prediction = loaded_model.predict_classes(img2)
            print("predicted digit:", str(prediction))
    else:
            print("Provide picture according to the specified parameters")

flag = True
while(True):
    choice = int(input(" 1 for prediction from web \n 2 for prediction from pc \n 3 for live prediction \n"))
    if(choice == 1):
        #https://previews.123rf.com/images/orson/orson1103/orson110300005/8996029-digital-number-four-check-my-portfolio-for-other-numbers-from-the-set.jpg
        #https://upload.wikimedia.org/wikipedia/commons/thumb/f/fc/3_number_black_and_white.svg/1003px-3_number_black_and_white.svg.png
        url = input('Enter the url')
        recognition(url)
        repeat = input("Do you want to recognize more digits(Y/N)")
        if (repeat == 'Y' or repeat == 'y'):
            pass
        else:
            break
    elif(choice == 2):
        path = input('Enter the path of image \n')
        recognition_pc(path)
        repeat = input("Do you want to recognize more digits(Y/N)")
        if (repeat == 'Y' or repeat == 'y'):
            pass
        else:
            break
    elif(choice == 3):
        print("Under maintainance\n")
        repeat = input("Do you want to recognize more digits(Y/N)")
        if (repeat == 'Y' or repeat == 'y'):
            pass
        else:
            break
    else:
        print("Enter Valid key")
