import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
from IPython.display import display
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


# prepare_data 函數和相關的數據處理
def prepare_data(data):
    """ Prepare data for modeling 
        input: data frame with labels and pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48, 1))
    image_label = np.array(list(map(int, data['emotion'])))

    # 定義 CLAHE 對象
    clahe = cv2.createCLAHE(clipLimit=2.0)

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))  # 將形狀調整為 (48, 48)
        # 應用 CLAHE
        image = clahe.apply(image.astype(np.uint8))
        image = np.expand_dims(image, axis=-1)  # 將圖像擴展為 (48, 48, 1) 的形狀
        image_array[i] = image  # 分配圖像到 image_array

    return image_array, image_label


def convert_to_3_channels(img_arrays):
    sample_size, nrows, ncols, c = img_arrays.shape
    img_stack_arrays = np.zeros((sample_size, nrows, ncols, 3))
    for _ in range(sample_size):
        img_stack = np.stack(
            [img_arrays[_][:, :, 0], img_arrays[_][:, :, 0], img_arrays[_][:, :, 0]], axis=-1)
        img_stack_arrays[_] = img_stack / 255
    return img_stack_arrays


# 測試
def evaluate_model(model, x_test, y_test, y_test_oh):
    emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    # 在测试集上评估模型
    results = model.evaluate(x_test, y_test_oh)

    # 打印评估结果
    print("損失（Loss）:", results[0])
    print("準確率（Accuracy）:", results[1])

    # 在测试集上生成预测并可视化结果
    y_true = []
    y_pred = []
    for i in range(len(x_test)):
        img = np.expand_dims(x_test[i], axis=0)
        label = y_test[i]
        pred = np.argmax(model.predict(img))

        y_true.append(label)
        y_pred.append(pred)

    unique_classes_true = np.unique(y_true)
    unique_classes_pred = np.unique(y_pred)

    print("y_true 中的唯一類別索引：", unique_classes_true)
    print("y_pred 中的唯一類別索引：", unique_classes_pred)

    # 随机选择一些图像进行可视化
    indices = np.random.choice(len(x_test), 20, replace=False)
    fig = plt.figure(figsize=(25, 4))
    for idx, index in enumerate(indices):
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        plt.imshow(x_test[index])
        ax.set_title("{} ({})".format(emotions[y_pred[index]], emotions[y_true[index]]),
                     color=("green" if y_pred[index] == y_true[index] else "red"))

    plt.show()

# 訓練
def train(model, epochs=30, batch_size=64):
    emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
                3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    # 資料切割(訓練、驗證、測試)
    df_raw = pd.read_csv("../fer2013.csv")
    df_train = df_raw[df_raw['Usage'] == 'Training']
    df_val = df_raw[df_raw['Usage'] == 'PublicTest']
    df_test = df_raw[df_raw['Usage'] == 'PrivateTest'].head(20)

    x_train, y_train = prepare_data(df_train)
    x_val, y_val = prepare_data(df_val)
    x_test, y_test = prepare_data(df_test)

    x_train = convert_to_3_channels(x_train)
    x_val = convert_to_3_channels(x_val)
    x_test = convert_to_3_channels(x_test)

    y_train_oh = to_categorical(y_train)
    y_val_oh = to_categorical(y_val)
    y_test_oh = to_categorical(y_test)

    # 比較各資料集大小
    display(df_raw['Usage'].value_counts())

    # Dara augmentation
    datagen = ImageDataGenerator()

    # class_weight
    class_sample_size = [np.where(y_train == c)[0].shape[0]
                         for c in range(len(emotions.keys()))]
    max_class_size = np.max(class_sample_size)
    class_weight = [max_class_size / size for size in class_sample_size]
    class_weight = dict(zip(emotions.keys(), class_weight))

    datagen.fit(x_train)

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 開始訓練
    history = model.fit(
        datagen.flow(x_train, y_train_oh, batch_size=batch_size),
        steps_per_epoch=len(x_train) // batch_size,
        validation_data=(x_val, y_val_oh),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        class_weight=class_weight,
        shuffle=True,  # 每個 epoch 都用不同的順序跑數據
        callbacks=[tensorboard_callback]  # 只添加 TensorBoard 回調函數
    )

    # 繪製損失曲線
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 繪製準確率曲線
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 訓練過程結束後，在測試集上評估模型
    evaluate_model(model, x_test, y_test, y_test_oh)

    # 只返回模型
    return model
