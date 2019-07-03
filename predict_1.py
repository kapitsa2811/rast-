from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join


# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('model_simpson.h5')

'''
    this is with more images
'''
model=load_model('/home/kapitsa/PycharmProjects/cartoon/datascience/machine_learning/dog_cat_classifier//model_batman_greenLantern_spiderman.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

mypath = "predict/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)
# predicting images
dog_counter = 0 
cat_counter  = 0
for file in onlyfiles:
    img = image.load_img(mypath+file, target_size=(img_width, img_height))
    print("\n\t shape=",img.size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    classes = classes[0][0]
    print("\n\t classes =",classes)
    if classes == 0:
        print(file + ": " + 'Batman')
        cat_counter += 1
        img.save("/home/kapitsa/PycharmProjects/cartoon/datascience/machine_learning/dog_cat_classifier/temp/temp/Batman/" + file)
    elif classes==1:
        img.save("/home/kapitsa/PycharmProjects/cartoon/datascience/machine_learning/dog_cat_classifier/temp/temp/GreenLantern/" + file)
        print(file + ": " + 'GreenLantern')
        dog_counter += 1
    else:
        img.save("/home/kapitsa/PycharmProjects/cartoon/datascience/machine_learning/dog_cat_classifier/temp/temp/Spiderman/" + file)
        print(file + ": " + 'Spiderman')
        dog_counter += 1


print("Total Dogs :",dog_counter)
print("Total Cats :",cat_counter)


