import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np
import os


X_train_path = "C:\\Users\\admin\\Downloads\\圖片\\tuxedo cat"
Y_train_path = "C:\\Users\\admin\\Downloads\\圖片\\灰階圖片\\orange cat_gray"

# 讀取 X 和 Y 的資料
def load_data(path):
    image_list = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(path, filename)
            with Image.open(image_path) as img:
                img = img.resize((640, 832))
                img_gray = img.convert('L')
                image_array = np.array(img_gray)
                image_list.append(image_array)
    return np.array(image_list)

X_train = load_data(X_train_path)
Y_train = load_data(Y_train_path)


# 建立模型
model = Sequential()

# 第一層卷積層
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(640, 832, 1)))

# 第二層卷積層
model.add(Conv2D(64, (3, 3), activation='relu'))

# 池化層
model.add(MaxPooling2D(pool_size=(2, 2)))

# 第三層卷積層
model.add(Conv2D(128, (3, 3), activation='relu'))

# 池化層
model.add(MaxPooling2D(pool_size=(2, 2)))

# 將特徵圖展平成一維向量
model.add(Flatten())

# 全連接層
model.add(Dense(64, activation='relu'))

# 輸出層
model.add(Dense(10, activation='softmax'))

# 編譯模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 顯示模型摘要
model.summary()

model.save("C:\\Users\\admin\\Downloads\\test")