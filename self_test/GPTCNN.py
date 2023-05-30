import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image
import numpy as np
import os

# 設定資料集路徑
dataset_path = "path_to_dataset_folder"

# 設定類別標籤
class_labels = ["花紋貓", "純色貓"]

# 讀取資料集並進行前處理
def load_dataset(path):
    image_list = []
    label_list = []
    for class_label, class_name in enumerate(class_labels):
        class_path = os.path.join(path, class_name)
        for filename in os.listdir(class_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(class_path, filename)
                with Image.open(image_path) as img:
                    img = img.resize((64, 64))  # 調整圖片大小
                    img_gray = img.convert('L')  # 轉為灰階
                    image_array = np.array(img_gray)
                    image_list.append(image_array)
                    label_list.append(class_label)
    return np.array(image_list), np.array(label_list)

# 載入並前處理資料集
X_train, Y_train = load_dataset(dataset_path)

# 正規化圖片資料
X_train = X_train / 255.0

# 建立模型
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(640, 832, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(class_labels), activation='softmax'))

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 訓練模型
model.fit(X_train, Y_train, epochs=10, batch_size=32)

# 儲存模型
model.save("cat_classification_model.h5")

# 測試圖片路徑
test_image_path = "path_to_test_image.jpg"

# 讀取測試圖片並進行前處理
with Image.open(test_image_path) as img:
    img = img.resize((64, 64))  # 調整圖片大小
    img_gray = img.convert('L')  # 轉為灰階
    test_image = np.array(img_gray)
    test_image = test_image.reshape((1, 64, 64, 1))  # 修改形狀以符合模型的預期輸入形狀
    test_image = test_image / 255
