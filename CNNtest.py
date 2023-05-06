import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

# 載入 CIFAR-10 資料集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 將像素值正規化到0到1之間
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 將標籤轉換為 one-hot 編碼
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# 建立 CNN 模型
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 編譯模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# 評估模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 使用模型對單張圖片進行預測
img = load_img('test_image.jpg', target_size=(32, 32))
img_array = img_to_array(img)
img_array = img_array.reshape((1,) + img_array.shape)
img_array = img_array / 255
pred = model.predict(img_array)
print('Predicted label:', np.argmax(pred))
