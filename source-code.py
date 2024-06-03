# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 05:54:02 2024

@author: kcv
"""


import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import PReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class KerasMember:
    def __init__(self, name, keras_model, train_batches, val_batches):
        self.name = name
        self.keras_model = keras_model
        self.train_batches = train_batches
        self.val_batches = val_batches

    def train(self, epochs=10):
        self.keras_model.fit(
            self.train_batches,
            epochs=epochs,
            validation_data=self.val_batches
        )

    def evaluate(self):
        return self.keras_model.evaluate(self.val_batches)

    def predict(self, test_batches):
        return self.keras_model.predict(test_batches)
    
"""Specifying global parameters"""

img_h,img_w= (300,300)
batch_size=128
epochs=10
n_class=3

"""*Concatenating train and test directory paths..*"""

base_dir = 'Mammogram-Splitted'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
validation_dir = os.path.join(base_dir, 'val')


from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

base_model_1=VGG19(include_top=False, weights='imagenet',input_shape=(img_h,img_w,3), pooling='avg')

# Making last layers trainable, because our dataset is much diiferent from the imagenet dataset
for layer in base_model_1.layers[:-6]:
    layer.trainable=False

model_1=Sequential()
model_1.add(base_model_1)

model_1.add(Flatten())
model_1.add(BatchNormalization())
model_1.add(Dropout(0.35))
model_1.add(Dense(n_class,activation='softmax'))

model_1.summary()


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
         rescale=1./255,
         rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


train_generator = train_datagen.flow_from_directory(
                    train_dir,                   # This is the source directory for training images
                    target_size=(img_h, img_w),  # All images will be resized to 300x300
                    batch_size=batch_size,
                    class_mode='categorical')

val_datagen= ImageDataGenerator(rescale=1./255)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
validation_generator = val_datagen.flow_from_directory(
                        validation_dir,
                        target_size=(img_h, img_w),
                        batch_size=batch_size,
                        class_mode='categorical')


#optimizer_2 = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
"""Compiling Models"""
model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Instantiate the KerasMember object
member1 = KerasMember(name="model1", keras_model=model_1, train_batches=train_generator, val_batches=validation_generator)

# Train the model
member1.train(epochs=10)

# Evaluate the model
evaluation_results = member1.evaluate()

print(f"Evaluation Results: {evaluation_results}")

# Predict using the model (assuming you have a test generator)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_h, img_w),  # adjust to your target size
    batch_size=batch_size,
    class_mode='categorical')

predictions = member1.predict(test_generator)
print(f"Predictions1: {predictions}")

"""Inception V3"""
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import ResNet50


base_model_2= InceptionV3(include_top=False, weights='imagenet',
                                        input_tensor=None, input_shape=(img_h,img_w,3), pooling='avg')

for layer in base_model_2.layers[:-30]:
    layer.trainable=False
model_2=Sequential()
model_2.add(base_model_2)
model_2.add(Flatten())
model_2.add(BatchNormalization())
model_2.add(Dense(1024,activation='relu'))
model_2.add(BatchNormalization())

model_2.add(Dense(512,activation='relu'))
model_2.add(Dropout(0.35))
model_2.add(BatchNormalization())

model_2.add(Dense(256,activation='relu'))
model_2.add(Dropout(0.35))
model_2.add(BatchNormalization())

model_2.add(Dense(n_class,activation='softmax'))

model_2.summary()

model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Instantiate the KerasMember object
member2 = KerasMember(name="model1", keras_model=model_2, train_batches=train_generator, val_batches=validation_generator)

# Train the model
member2.train(epochs=10)

# Evaluate the model
evaluation_results = member2.evaluate()


print(f"Evaluation Results: {evaluation_results}")

# Predict using the model (assuming you have a test generator)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_h, img_w),  # adjust to your target size
    batch_size=batch_size,
    class_mode='categorical')

predictions_2 = member2.predict(test_generator)
print(f"Predictions2: {predictions_2}")

from deepstack.ensemble import DirichletEnsemble
from sklearn.metrics import accuracy_score


wAvgEnsemble = DirichletEnsemble(N=10000, metric=accuracy_score)
wAvgEnsemble.add_members([member1, member2])
wAvgEnsemble.fit(train_generator,validation_generator)
wAvgEnsemble.describe()


from deepstack.ensemble import StackEnsemble
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

#Ensure you have the scikit-learn version >= 0.22 installed
print("sklearn version must be >= 0.22. You have:", sklearn.__version__)

stack = StackEnsemble()

# 2nd Level Meta-Learner
estimators = [
    ('rf', RandomForestClassifier(verbose=0, n_estimators=100, max_depth=15, n_jobs=20, min_samples_split=30)),
    ('etr', ExtraTreesClassifier(verbose=0, n_estimators=100, max_depth=10, n_jobs=20, min_samples_split=20)),
    ('dtc',DecisionTreeClassifier(random_state=0, max_depth=3))
]
# 3rd Level Meta-Learner
clf = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression()
)

stack.model = clf
stack.add_members([member1, member2])
stack.fit()
stack.describe(metric=sklearn.metrics.accuracy_score)

"""Now lets save our stack-ensemble model.

Saves meta-learner and base-learner of ensemble into folder / directory.

**Args:**
* folder: the folder where models should be saved to(Create if not exists).
"""

stack.save()

"""Loading the model is as simple as this."""

stack.load()

"""Don't forget to save the DCNN itself, otherwise you may have trouble."""

model_json = model_1.to_json()
with open("VGG_19.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_1.save_weights("VGG_19_weights.h5")
print("Saved VGG19 to disk")

model_json = model_2.to_json()
with open("Inception_V3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_2.save_weights("Inception_V3_weights.h5")
print("Saved Inception_V3 to disk")


