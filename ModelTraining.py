import pandas as pd
import numpy as np
import tensorflow as tf
from IPython.display import display
from tensorflow.keras.utils import to_categorical
import mlflow.tensorflow
from keras.preprocessing.image import ImageDataGenerator


# prepare_data 函數和相關的數據處理
def prepare_data(data):
    """ Prepare data for modeling 
        input: data frame with labels and pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48, 1))
        image_array[i] = image

    return image_array, image_label


def convert_to_3_channels(img_arrays):
    sample_size, nrows, ncols, c = img_arrays.shape
    img_stack_arrays = np.zeros((sample_size, nrows, ncols, 3))
    for _ in range(sample_size):
        img_stack = np.stack(
            [img_arrays[_][:, :, 0], img_arrays[_][:, :, 0], img_arrays[_][:, :, 0]], axis=-1)
        img_stack_arrays[_] = img_stack / 255
    return img_stack_arrays


# 訓練＋評估
def train(model_mobilenet, name="logs", epochs=30, batch_size=32):
    # 資料切割(訓練、驗證、測試)
    df_raw = pd.read_csv("fer2013.csv")
    df_train = df_raw[df_raw['Usage'] == 'Training']
    df_val = df_raw[df_raw['Usage'] == 'PublicTest']
    df_test = df_raw[df_raw['Usage'] == 'PrivateTest']

    X_train, y_train = prepare_data(df_train)
    X_val, y_val = prepare_data(df_val)
    X_test, y_test = prepare_data(df_test)

    X_train = convert_to_3_channels(X_train)
    X_val = convert_to_3_channels(X_val)
    X_test = convert_to_3_channels(X_test)

    y_train_oh = to_categorical(y_train)
    y_val_oh = to_categorical(y_val)
    y_test_oh = to_categorical(y_test)

    # 比較各資料集大小
    display(df_raw['Usage'].value_counts())

    # 測試模型輸入與輸出是否符合資料格式
    prob_mobilenet = model_mobilenet(X_train[:1]).numpy()
    print(prob_mobilenet.shape)

    # Check if there's an active run, and end it if necessary
    if mlflow.active_run():
        mlflow.end_run()

    # 記錄模型結構
    with mlflow.start_run():
        mlflow.tensorflow.autolog()  # 記錄參數

        # 可視化訓練過程
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=name)

        # dara argumentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        datagen.fit(X_train)

        # 開始訓練(要放在mlflow start_run內進行)
        hist_mobilenet = model_mobilenet.fit(
            datagen.flow(X_train, y_train_oh, batch_size=batch_size),
            steps_per_epoch=len(X_train) / batch_size,
            validation_data=(X_val, y_val_oh),
            epochs=epochs, 
            batch_size=batch_size,
            verbose=1,
            shuffle=True,  # 每個epoch都用不同的順序跑數據
            callbacks=[tensorboard_callback]
        )

        # 評估模型
        results = model_mobilenet.evaluate(X_test, y_test_oh)

        # 印出評估結果
        print("損失（Loss）:", results[0])
        print("準確率（Accuracy）:", results[1])

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)

        # 測試記錄
        mlflow.log_metric("test_loss", results[0])
        mlflow.log_metric("test_accuracy", results[1])