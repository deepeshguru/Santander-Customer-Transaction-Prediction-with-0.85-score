from __future__ import absolute_import, division, print_function

import os
import numpy as np
import pandas as pd


import tensorflow as tf
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"train.csv")
df = df.sample(frac=1)

#df0 = df[df["target"]==0]
#df0 = df.sample(frac=0.33)

#df1 = df[df["target"] == 1]

#dff = pd.concat([df0, df1])
#dff = dff.sample(frac=1)


X = df.iloc[:, 2:].values
Y = df.iloc[:, 1].values
k = 0.1
xt, xtt, yt, ytt = train_test_split(X, Y, test_size=0.05, random_state=42)
i=1000
j = tf.nn.relu
k=0.4
model = tf.keras.models.Sequential([tf.keras.layers.Dense(i,input_shape=(200,), activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(i, activation=j),
                                    tf.keras.layers.Dropout(rate=k),
                                    tf.keras.layers.BatchNormalization(),
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(xt, yt, batch_size=380, validation_data=(xtt, ytt), epochs=100, use_multiprocessing=True)

test_loss, test_acc = model.evaluate(xtt, ytt, batch_size=200, use_multiprocessing=True)
model.save("santand1.h5")

print('Test accuracy:', test_acc)
print("parameter:" ,str(i), str(j), str(k))
