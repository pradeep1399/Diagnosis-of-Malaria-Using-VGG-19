# VGG-19 model for Diagnosis of Malaria using blood cells
# @pradeep July 2021


###############################################################################################################
from google.colab import drive
drive.mount('/content/drive')

###############################################################################################################
import tensorflow as tf
tf.test.is_gpu_available()

###############################################################################################################
# import the libraries as shown below
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

###############################################################################################################
# re-size all the images to this
IMAGE_SIZE = [224, 224]

###############################################################################################################
train_path = '.../drive/MyDrive/cell/train'
valid_path = '.../drive/MyDrive/cell/val'

###############################################################################################################
import cv2
images=cv2.imread(".../MyDrive/cell/train/Parasitized/C68P29N_ThinF_IMG_20150819_134326_cell_154.png")
plt.imshow(images)
plt.show()

###############################################################################################################
# Import the Vgg 19 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

###############################################################################################################
mobilnet = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

###############################################################################################################
# don't train existing weights
for layer in mobilnet.layers:
    layer.trainable = False

###############################################################################################################
# useful for getting number of output classes
folders = glob('.../drive/MyDrive/cell/train/*')
folders

###############################################################################################################
# our layers - you can add more if you want
x = Flatten()(mobilnet.output)

###############################################################################################################
prediction = Dense(len(folders), activation='softmax')(x)

###############################################################################################################
# create a model object
model = Model(inputs=mobilnet.input, outputs=prediction)
# view the structure of the model
model.summary()

###############################################################################################################
from tensorflow.keras.layers import MaxPooling2D
### Create Model 
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()

###############################################################################################################
# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

###############################################################################################################
# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
val_datagen = ImageDataGenerator(rescale = 1./255)

###############################################################################################################
# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('.../drive/MyDrive/cell/train',
                                                 target_size = (224, 224),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

###############################################################################################################
training_set

###############################################################################################################
val_set = val_datagen.flow_from_directory('.../drive/MyDrive/cell/val',
                                            target_size = (224, 224),
                                            batch_size = 64,
                                            class_mode = 'categorical')

###############################################################################################################
# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=val_set,
  epochs=30,
  steps_per_epoch=len(training_set),
  validation_steps=len(val_set)
)

###############################################################################################################
# save it as a h5 file
from tensorflow.keras.models import load_model
model.save('.../drive/MyDrive/model_vgg19.h5')

###############################################################################################################
# plot the loss
# naming the x axis
plt.xlabel('epoch')
# naming the y axis
plt.ylabel('loss')
# giving a title to my graph
plt.title('train_loss & val_loss')
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
# naming the x axis
plt.xlabel('epoch')
# naming the y axis
plt.ylabel('accuracy')
# giving a title to my graph
plt.title('train_accuracy & val_accuracy')
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

###############################################################################################################
y_pred = model.predict(val_set)
y_pred

###############################################################################################################
import numpy as np
y_pred = np.argmax(y_pred, axis=1)


###############################################################################################################
#prediction of malaria cell 
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
#load model
model=load_model('/content/drive/MyDrive/model_vgg19.h5')
from tensorflow.keras.preprocessing import image
img = image.load_img('..../Parasitized/C48P9thinF_IMG_20150721_160406_cell_226.png',target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
result = model.predict(img)
if result[0][1] == 1.0:
  prediction = 'Uninfected image:  person is not infected' 
else:
  prediction = 'Parasitized image:  person is infected' 

print(prediction)

images=cv2.imread("..../Parasitized/C48P9thinF_IMG_20150721_160406_cell_226.png")
plt.imshow(images)
plt.show()
