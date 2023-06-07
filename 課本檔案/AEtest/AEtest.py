import matplotlib.pyplot as plt
import numpy as np
import tensorflow
from keras.datasets import mnist    #手寫數字資料
from keras.models import Model
from keras.layers import Dense, Input

#載入minst
(X_train,_),(X_test,_) = mnist.load_data()

#圖片大小轉換成28*28
X_train = X_train.reshape(X_train.shape[0], 28*28)
#圖片類型轉成浮點數
X_train = X_train.astype("float32")
#輸出
print("X_train Shape:",X_train.shape)

#圖片大小轉換成28*28
X_test = X_test.reshape(X_test.shape[0], 28*28)
#圖片類型轉成浮點數
X_test = X_test.astype("float32")
#輸出
print("X_test Shape:",X_test.shape)

#正規化數值(介於0~1)
X_train = X_train/255
X_test = X_test/255

#建立網路

#輸入層(784layer)
image = Input(shape=(784,))
#第一隱含層(屬於encoder)    encoder=>解碼器,將影像重點挑出
First = Dense(300, activation="relu")(image)
#第二隱含層(屬於辨識集)     encoder的影像重點
Second = Dense(144,activation="relu")(First)
#第三隱含層(屬於decoder)    decoder=>編碼器,將影像重現
Third = Dense(300,activation="relu")(Second)
#輸出層(784),資料型態為array
image_out = Dense(784,activation="sigmoid")(Third)

#AE
ae = Model(image, image_out)
ae.summary()

#encoder
encoder = Model(image, Second)
encoder.summary()

#decoder
decoder_img = Input(shape=(144,))
layer1 = ae.layers[-2](decoder_img)
layer2 = ae.layers[-1](layer1)
decoder = Model(decoder_img, layer2)
decoder.summary()

ae.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

#訓練模型
ae.fit(X_train,X_train,validation_data=(X_test,X_test), epochs=10, batch_size=128, verbose=2)

#做預判()
encoded = encoder.predict(X_test)
decoded = decoder.predict(encoded)

#設定輸出畫面大小(10:3)
plt.figure(figsize=(10, 3))
#設定顯示預測出張數8張
x = 8
for i in range(x):

    y = plt.subplot(3, 8, i + 1)
    #顯示28*28原始圖片
    y.imshow(X_test[i].reshape(28, 28), cmap="binary")
    y.axis("off")

    y = plt.subplot(3, 8, i + 1 + x)
    #顯示12*12辨識集圖片
    y.imshow(encoded[i].reshape(12, 12), cmap="binary")
    y.axis("off")

    y = plt.subplot(3, 8, i + 1 + 2 * x)
    #顯示28*28預判圖片
    y.imshow(decoded[i].reshape(28, 28), cmap="binary")
    y.axis("off")