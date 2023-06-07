import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input

(X_train,_),(X_test,_) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 28*28)
X_train = X_train.astype("float32")
print("X_train Shape:",X_train.shape)

X_test = X_test.reshape(X_test.shape[0], 28*28)
X_test = X_test.astype("float32")
print("X_test Shape:",X_test.shape)

X_train = X_train/255
X_test = X_test/255

image = Input(shape=(784,))
First = Dense(300, activation="relu")(image)
Second = Dense(144,activation="relu")(First)
Third = Dense(300,activation="relu")(Second)
image_out = Dense(784,activation="sigmoid")(Third)

ae = Model(image, image_out)
ae.summary()

encoder = Model(image, Second)
encoder.summary()

decoder_img = Input(shape=(144,))
layer1 = ae.layers[-2](decoder_img)
layer2 = ae.layers[-1](layer1)
decoder = Model(decoder_img, layer2)
decoder.summary()

ae.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

ae.fit(X_train,X_train,validation_data=(X_test,X_test), epochs=10, batch_size=128, verbose=2)

encoded = encoder.predict(X_test)
decoded = decoder.predict(encoded)

plt.figure(figsize=(10, 3))
x = 8
for i in range(x):

    y = plt.subplot(3, 8, i + 1)
    y.imshow(X_test[i].reshape(28, 28), cmap="binary")
    y.axis("off")

    y = plt.subplot(3, 8, i + 1 + x)
    y.imshow(encoded[i].reshape(12, 12), cmap="binary")
    y.axis("off")

    y = plt.subplot(3, 8, i + 1 + 2 * x)
    y.imshow(decoded[i].reshape(28, 28), cmap="binary")
    y.axis("off")