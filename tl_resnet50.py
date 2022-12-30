import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from tensorflow.keras.applications.resnet50 import ResNet50

classes = {'no':0, 'yes':1}
X = []
y = []
for i in classes:
    path = './input/brain_tumor_dataset/'+i
    for j in os.listdir(path):
        img = cv2.imread(path+'/'+j)        
        img = cv2.resize(img, (128,128))
        (b, g, r)=cv2.split(img) 
        img=cv2.merge([r,g,b])
        X.append(img)
        y.append(classes[i])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train = X_train.reshape(-1, 128, 128, 3)
X_test = X_test.reshape(-1, 128, 128, 3)

resnet_model= ResNet50(input_shape=(128,128,3), weights='imagenet', include_top=False)
model=tf.keras.models.Sequential()
model.add(resnet_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

for layer in resnet_model.layers:
    layer.trainable = False

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=X_train, y=y_train, epochs=10)

y_predict =model.predict(X_test).reshape(-1)
acc=keras.metrics.binary_accuracy(y_test, y_predict, threshold=0.5)
print("accuracy:",acc)

y_predict = [1 if i>=0.5 else 0 for i in y_predict]
print(classification_report(y_predict,y_test))

# to visualize the model architecture.
#import visualkeras
#visualkeras.layered_view(model, legend=True).save('tl_resnet50_model.png')

