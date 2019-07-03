from keras.models import load_model
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
#from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.models import  Sequential
from keras.layers import Dense,Dropout,Flatten
from keras import optimizers,regularizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import losses
from keras.preprocessing import image
#from keras.datasets import mnist
import numpy as np
import pandas as pd
import os
#import numpy as np


model1=load_model("/home/kapitsa/PycharmProjects/cartoon/model//model.h5")
model1.compile(loss='categorical_crossentropy', optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])

print("\n\t model summary=",model1.summary())

'''resolution=224
batch_size=20

data_datagen_e=ImageDataGenerator(rescale=1./255,validation_split=0.15)

try:
    testDataGenerator = data_datagen_e.flow_from_directory(
        "//home/kapitsa/PycharmProjects/cartoon/dataset/test//",
        classes=['homer_simpson', 'ned_flanders'],
        target_size=(resolution, resolution),
        batch_size=batch_size,
        class_mode='categorical', subset="training")

    predict=model1.predict_generator(testDataGenerator)
    print("\n\t predict=",predict)
except Exception as e:
    print("\n\t e=",e)
    pass
'''
import os
import cv2
import numpy as np

path="//home/kapitsa/PycharmProjects/cartoon/dataset/test//"
folder=os.listdir(path)

print("\n\t folder=",folder)

homer=[]
ned=[]


for indx,name  in enumerate(os.listdir(path+folder[0])):

    #print("\n\t indx=",indx,"\t image name=",name)
    image=cv2.imread(path+folder[0]+"//"+name)
    image=cv2.resize(image,(224,224))
    homer.append(image)

    #print("\n\t shape=",image.shape)


for indx,name  in enumerate(os.listdir(path+folder[1])):
     #print("\n\t indx=",indx,"\t image name=",name)
     image = cv2.imread(path + folder[1] + "//" + name)
     image = cv2.resize(image, (224, 224))
     ned.append(image)
     #print("\n\t shape=", image.shape)


print("\n\t homer=",len(homer))
print("\n\t ned =",len(ned))

homer=np.array(homer)
ned=np.array(ned)


for i,temp in enumerate(homer):
    print("\n\t i=",i)
    temp = np.expand_dims(temp, axis=0)

    v=model1.predict(np.array(temp))
    print("\n\t v=",len(v[0]))
    print("\n\t v=",v[0])
    #input("check")


for i, temp in enumerate(ned):
    print("\n\t i=", i)
    temp = np.expand_dims(temp, axis=0)
    v = model1.predict(temp)
    print("\n\t v=", v)

    input("check")


# print("\n\t homer=",np.array(homer))
# print("\n\t ned =",np.array(ned))


'''
    prediction
'''
'''
h=model1.predict(homer)
print("\n\t h=",h)

v = model1.predict(ned)
print("\n\t v=",v)
'''
