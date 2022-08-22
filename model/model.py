from keras.models import Sequential
from sklearn.model_selection import train_test_split
from PIL import Image

import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from torchvision import datasets
import glob


from keras.layers import Input, Conv2D, Activation, MaxPool2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


categories = []

for i in range(1, 101):
    categories.append(f"{i}")

print(categories)



data_dir = "/workspace/2022_ai_study/data"
num_classes = len(categories)

image_w = 224
image_h = 224

pixels = image_w * image_h * 3
 
x = []
y = []

for idx, category in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idx] = 1

    image_dir = data_dir + "/" + category
    files = glob.glob(image_dir + "/*.jpg") 

    for i,f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)

        x.append(data)
        y.append(label)

        if i % 1000 == 0:
            print(category, " : ", f)
            

x = np.array(x)
y = np.array(y)
    
x_train, x_test, y_train, y_test = train_test_split(x, y)
xy = (x_train, x_test, y_train, y_test)

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


from keras.layers import Input, Conv2D, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam 
from keras.models import Model

inputlayer = Input(shape=(224, 224, 3), dtype='float32')

cnn1 = Conv2D(128, kernel_size=3, activation='relu')(inputlayer)
cnn1 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn1 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn1 = MaxPool2D(pool_size=3, strides=2)(cnn1)

cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn1)
cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn2 = Conv2D(128, kernel_size=3, activation='relu')(cnn2)
cnn2 = MaxPool2D(pool_size=3, strides=2)(cnn2)

cnn3 = Conv2D(256, kernel_size=3, activation='relu')(cnn2)
cnn3 = Conv2D(256, kernel_size=3, activation='relu')(cnn3)
cnn3 = Conv2D(256, kernel_size=3, activation='relu')(cnn3)
cnn3 = MaxPool2D(pool_size=3, strides=2)(cnn3)

cnn4 = Conv2D(512, kernel_size=3, activation='relu')(cnn3)
cnn4 = Conv2D(512, kernel_size=3, activation='relu')(cnn4)
cnn4 = Conv2D(512, kernel_size=3, activation='relu')(cnn4)
cnn4 = MaxPool2D(pool_size=3, strides=2)(cnn4)

dense = Flatten()(cnn4)
dense = Dropout(0.2)(dense)
dense = Dense(1024, activation='relu')(dense)
dense = Dense(1024, activation='relu')(dense)

output = Dense(1, activation='linear', name='age')(dense)

model = Model(input, output)
model.compile(optimizer=Adam(0.0001), loss='mse', metrics=['mae'])
model.summary()
    

model.fit(x_train, y_train, batch_size=30, epochs=5000)

model_json = additional_model.to_json()
with open("model.json", "w") as json_file : 
    json_file.write(model_json)