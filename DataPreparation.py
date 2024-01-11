import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import display



# CSV 文件，讀入 DataFrame
df_raw = pd.read_csv('fer2013.csv')
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
             3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

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

# 使用 prepare_data 函數
X_train, y_train = prepare_data(df_raw[df_raw['Usage'] == 'Training'])
X_val, y_val = prepare_data(df_raw[df_raw['Usage'] == 'PublicTest'])
X_test, y_test = prepare_data(df_raw[df_raw['Usage'] == 'PrivateTest'])


# 比較各資料集大小
display(df_raw['Usage'].value_counts())
# 8:1:1
# output:
# Training       28709
# PrivateTest     3589
# PublicTest      3589
# Name: Usage, dtype: int64


# 觀察每一類的表情圖片(個抽7張) 
def plot_one_emotion(data, img_arrays, img_labels, label=0):
    fig, axs = plt.subplots(1, 7, figsize=(25, 12))# fig是整個圖形基礎,創建一個包含七個子圖的畫布，並設定圖像的大小。axs 是包含了 7 個子圖的一維陣列，表示一行七列的子圖
    fig.subplots_adjust(hspace=.2, wspace=.2)# 設定子圖之間的間距
    axs = axs.ravel()

    for i in range(7):#每個子畫布裡子圖的數量
        idx = data[data['emotion'] == label].index[i]#取得滿足條件的那一列在整個樣本中的索引
        axs[i].imshow(img_arrays[idx][:, :, 0], cmap='gray')# 使用 axs[i].imshow 顯示圖像，img_arrays[idx][:, :, 0] 表示灰階圖像的像素數據
        axs[i].set_title(emotions[img_labels[idx]])
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])
        
# for label in emotions.keys():
    plot_one_emotion(df_raw, X_train, y_train, label=label)


# 觀察表情分布
def plot_distributions(img_labels_1, img_labels_2, title1='', title2=''):
    df_array1 = pd.DataFrame()
    df_array2 = pd.DataFrame()
    df_array1['emotion'] = img_labels_1
    df_array2['emotion'] = img_labels_2

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    x = emotions.values()

    y = df_array1['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0    
    bars1 = axs[0].bar(x, y.sort_index(), color='orange')
    axs[0].set_title(title1)
    axs[0].grid()

    y = df_array2['emotion'].value_counts()
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    bars2 = axs[1].bar(x, y.sort_index())
    axs[1].set_title(title2)
    axs[1].grid()

     # 在每個柱子頂部顯示具體數量
    for bar in bars1:
        yval = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 1), ha='center', va='bottom')

    for bar in bars2:
        yval = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 1), ha='center', va='bottom')


    plt.show()
    
plot_distributions(
    y_train, y_val, title1='train labels', title2='val labels')
    

