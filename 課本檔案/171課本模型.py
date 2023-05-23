import tensorflow as tf
import pandas as pd
from tensorflow import saved_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical


(X_train,Y_train),(X_test,Y_test)=mnist.load_data()

X_train=X_train.reshape(X_train.shape[0],28,28,1)
X_train=X_train.astype("float32")

X_test=X_test.reshape(X_test.shape[0],28,28,1)
X_test=X_test.astype("float32")

X_train=X_train/255
X_test=X_test/255

Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)

model=Sequential()
model.add(Conv2D(8,kernel_size=(5,5),padding="same",input_shape=(28,28,1),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,kernel_size=(5,5),padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64,activation="relu"))
model.add(Dense(10,activation="softmax"))
model.summary()

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

history=model.fit(X_train,Y_train,validation_split=0.2,epochs=10,batch_size=128,verbose=2)

loss,accuracy=model.evaluate(X_train,Y_train)
print("訓練資料精準度 = {:.2f}".format(accuracy))
loss,accuracy=model.evaluate(X_test,Y_test)
print("測試資料精準度 = {:.2f}".format(accuracy))

model.save('C:\\Users\\admin\\Downloads\\my_model')
