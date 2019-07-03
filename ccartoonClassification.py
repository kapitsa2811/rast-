import keras

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
#import os

WEIGHTS_PATH_NO_TOP ="/home/kapitsa/PycharmProjects/cartoon/dataset/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"

def modelArchitecture():
    print("**")
    regularizer_weight = 0.001
    #model_resnet=ResNet50(include_top=False,weights="imagenet", input_shape=(224,224,3))
    #model_resnet=VGG16(include_top=False, input_shape=(224,224,3))
    #x=model_resnet.output

    model_resnet = resnet50.ResNet50(include_top=False,weights="imagenet", input_shape=(224,224,3))
    model=Sequential()
    model.add(model_resnet)
    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(2,activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',optimizer=optimizers.adagrad(lr=0.001), metrics=['acc'])
    print("\n\t model summary=", model.summary())
    return model

model=modelArchitecture()

print("\n\t model summary=",model.summary())

resolution=224
batch_size=20

data_datagen_e=ImageDataGenerator(rescale=1./255,validation_split=0.15)
#train_generator_e=data_datagen_e.flow_from_directory()

train_generator_e = data_datagen_e.flow_from_directory(
         "/home/kapitsa/PycharmProjects/cartoon/dataset/simpsons_dataset//",
         # classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson',
         #          'bart_simpson', 'marge_simpson', 'krusty_the_clown',
         #          'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten'],
         classes=['homer_simpson', 'ned_flanders'],
         target_size=(resolution, resolution),
         batch_size=batch_size,
         class_mode='categorical', subset="training")


val_generator_e = data_datagen_e.flow_from_directory(
        "/home/kapitsa/PycharmProjects/cartoon/dataset/simpsons_dataset//",
        # classes=['homer_simpson', 'ned_flanders', 'moe_szyslak', 'lisa_simpson',
        #          'bart_simpson', 'marge_simpson', 'krusty_the_clown',
        #          'principal_skinner', 'charles_montgomery_burns', 'milhouse_van_houten'],
        classes=['homer_simpson', 'ned_flanders'],
        target_size=(resolution, resolution),
        batch_size=batch_size,
        class_mode='categorical', subset="validation")


historye = model.fit_generator(
        train_generator_e,
        steps_per_epoch=(11745 // batch_size),
        epochs=10,
        validation_data=val_generator_e,
        validation_steps=(2066 // batch_size)
    )

model.save("/home/kapitsa/PycharmProjects/cartoon/model//model.h5")
print("Saved model to disk")

from keras.models import load_model

model1=load_model("/home/kapitsa/PycharmProjects/cartoon/model//model.h5")
print("\n\t model summary=",model1.summary())

try:
    testDataGenerator = data_datagen_e.flow_from_directory(
        "//home/kapitsa/PycharmProjects/cartoon/dataset/test/ned_flanders//",
        classes=['homer_simpson', 'ned_flanders'],
        target_size=(resolution, resolution),
        batch_size=batch_size,
        class_mode='categorical', subset="training")

    predict=model1.predict_generator(testDataGenerator)
    print("\n\t predict=",predict)
except Exception as e:
    print("\n\t e=",e)
    pass




#print("\n\t eval=",model.evaluate(val_generator_e))

'''
acc = historye.history['acc']
val_acc = historye.history['val_acc']
loss = historye.history['loss']
val_loss = historye.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
'''










