import cv2
import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

classes = {'no':0, 'yes':1}
X = []
y = []
for i in classes:
    path = './input/brain_tumor_dataset/'+i
    for j in os.listdir(path):
        img = cv2.imread(path+'/'+j, 0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(img, (128,128),interpolation=cv2.INTER_CUBIC)
        img = img / 255 
        X.append(img)
        y.append(classes[i])

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train = X_train.reshape(-1, 128, 128, 1)
X_test = X_test.reshape(-1, 128, 128, 1)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32,
                           kernel_size=(4,4),
                           activation='relu',
                           input_shape=(128,128,1)),
    tf.keras.layers.Conv2D(16,(4,4),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),
                              padding='valid'),
    tf.keras.layers.Conv2D(32,(4,4),activation='relu'),
    tf.keras.layers.Conv2D(16,3,activation='relu'),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

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
#visualkeras.layered_view(model, legend=True).save('cnn_model.png')