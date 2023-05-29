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

# 訓練模型
model.fit(X_train, Y_train, epochs=10, batch_size=32)

model.save("C:\\Users\\admin\\Downloads\\test")

test_image_path = "C:\\Users\\admin\\Downloads\\圖片\\灰階圖片\\tuxedo cat_gray\\DreamShaper_32_whitebackgroundtuxedo_cat_0 (2).jpg"
with Image.open(test_image_path) as img:
    img = img.resize((832, 640))  # 調整圖片大小
    img_gray = img.convert('L')
    test_image = np.array(img_gray)
    test_image = test_image.reshape((1, 640, 832))  # 修改形狀以符合模型的預期輸入形狀


# 預測
predictions = model.predict(test_image)
predicted_label = np.argmax(predictions)  # 根據預測結果找到最高機率的類別

# 印出預測結果
print('Predicted label:', predicted_label)

# 如果有真實標籤，可以進行評估
true_label = 0  # 真實標籤的數值，根據你的測試圖片而定
accuracy = predictions[0][true_label]  # 根據真實標籤的索引獲取預測的準確性
print('Accuracy:', accuracy)