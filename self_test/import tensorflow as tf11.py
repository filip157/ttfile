import tensorflow as tf
from PIL import Image
import numpy as np


# 載入模型
model = tf.keras.models.load_model("C:\\Users\\admin\\Downloads\\test")

# 載入測試圖片
test_image_path = "C:\\Users\\admin\\Downloads\\圖片\\灰階圖片\\tuxedo cat_gray\\DreamShaper_32_whitebackgroundtuxedo_cat_0 (2).jpg"
with Image.open(test_image_path) as img:
    img = img.resize((832, 640))  # 調整圖片大小
    img_gray = img.convert('L')
    test_image = np.array(img_gray)
    test_image = test_image.reshape((1, 640, 832, 1))  # 修改形狀以符合模型的預期輸入形狀


# 預測
predictions = model.predict(test_image)
predicted_label = np.argmax(predictions)  # 根據預測結果找到最高機率的類別

# 印出預測結果
print('Predicted label:', predicted_label)

# 如果有真實標籤，可以進行評估
true_label = 0  # 真實標籤的數值，根據你的測試圖片而定
accuracy = predictions[0][true_label]  # 根據真實標籤的索引獲取預測的準確性
print('Accuracy:', accuracy)