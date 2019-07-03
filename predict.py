from keras.models import  load_model
from keras.preprocessing.image import ImageDataGenerator

resolution=224,
batch_size=20
model=load_model("/home/kapitsa/PycharmProjects/cartoon/model//model.h5")
print("\n\t model summary=",model.summary())

data_datagen_e=ImageDataGenerator(rescale=1./255)


'''
def preprocess_fn(im_array):
    return im_array / 255 - .5


class foo(object):
    pass


#imgen = ImageDataGenerator(preprocessing_function='pass anything')

dataGenerator.image_data_generator = foo
dataGenerator.image_data_generator.preprocessing_function = preprocess_fn

#flowgen = imgen.flow_from_directory('/tmp/training_folder/')


testDataGenerator=dataGenerator.flow_from_directory( "/home/kapitsa/PycharmProjects/cartoon/dataset/simpsons_dataset//",
         classes=['homer_simpson', 'ned_flanders'],
         target_size=(resolution, resolution),
         batch_size=batch_size,
         class_mode='categorical', subset="training")


predict=model.predict_generator(testDataGenerator)
'''

try:
    testDataGenerator = data_datagen_e.flow_from_directory(
        "/home/kapitsa/PycharmProjects/cartoon/dataset/simpsons_dataset//",
        classes=['homer_simpson', 'ned_flanders'],
        target_size=(resolution, resolution),
        batch_size=batch_size,
        class_mode='categorical', subset="training")

    predict=model.predict_generator(testDataGenerator)
    print("\n\t predict=",predict)
except Exception as e:
    print(e)
    pass



