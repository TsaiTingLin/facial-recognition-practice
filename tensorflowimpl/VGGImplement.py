
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import pandas as pd
import numpy as np
import mlflow



# 接下來是 prepare_data 函數和相關的數據處理
def prepare_data(data):
    """ Prepare data for modeling 
        input: data frame with labels and pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48, 1))# Q
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')# 數據是以空格分隔的數字字符串，使用 np.fromstring 函數將其轉換為數字數組numpy array
        image = np.reshape(image, (48, 48, 1))  # 灰階圖的channel數為1
        image_array[i] = image

    return image_array, image_label

# VGG預訓練模型的輸入層需要3通道的圖像，所以我直接複製第1通道的陣列到2和3通道上
def convert_to_3_channels(img_arrays):
    sample_size, nrows, ncols, c = img_arrays.shape
    img_stack_arrays = np.zeros((sample_size, nrows, ncols, 3))
    for _ in range(sample_size):
        img_stack = np.stack(
            [img_arrays[_][:, :, 0], img_arrays[_][:, :, 0], img_arrays[_][:, :, 0]], axis=-1)
        img_stack_arrays[_] = img_stack/255
    return img_stack_arrays

# 建立預訓練模型
def build_model(preModel=VGG16, num_classes=7):
    pred_model = preModel(include_top=False, weights='imagenet',
                              input_shape=(48, 48, 3),
                              pooling='max', classifier_activation='softmax')
    output_layer = Dense(
        num_classes, activation="softmax", name="output_layer")

    model = tf.keras.Model(
        pred_model.inputs, output_layer(pred_model.output))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    return model



df_raw = pd.read_csv("../fer2013.csv")
# 資料切割(訓練、驗證、測試)
df_train = df_raw[df_raw['Usage'] == 'Training']
df_val = df_raw[df_raw['Usage'] == 'PublicTest']

X_train, y_train = prepare_data(df_train)
X_val, y_val = prepare_data(df_val)

X_train = convert_to_3_channels(X_train)
X_val = convert_to_3_channels(X_val)

y_train_oh = to_categorical(y_train)
y_val_oh = to_categorical(y_val)


# 測試模型輸入與輸出是否符合資料格式
model_vgg16 = build_model()
prob_vgg16 = model_vgg16(X_train[:1]).numpy()
print(prob_vgg16.shape)

# 開始訓練
epochs = 10
batch_size = 32
# 記錄參數
mlflow.tensorflow.autolog()

hist1 = model_vgg16.fit(X_train, y_train_oh, validation_data=(X_val, y_val_oh),
                        epochs=epochs, batch_size=batch_size)