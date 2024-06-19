import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

# 檢查MPS是否可用
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class FER2013Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 提取圖像像素和標籤
        pixels = self.dataframe.iloc[idx, 1].split()  # 將像素字符串分割成單個像素值
        image = np.array(pixels, dtype=np.uint8).reshape(48, 48)  # 將像素轉換為48x48的NumPy數組
        image = Image.fromarray(image)  # 將NumPy數組轉換為PIL圖像
        image = image.convert('RGB')  # 將灰度圖像轉換為RGB圖像（3通道）

        if self.transform:
            image = self.transform(image)

        label = int(self.dataframe.iloc[idx, 0])
        return image, label


# 定義情緒標籤字典
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# 讀取數據集
df_raw = pd.read_csv('../fer2013.csv')

# 按Usage字段劃分數據
train_data = df_raw[df_raw['Usage'] == 'Training']
val_data = df_raw[df_raw['Usage'] == 'PublicTest']
test_data = df_raw[df_raw['Usage'] == 'PrivateTest']

# 選擇每個情緒類別前1000個樣本來構建訓練集
train_data_1000 = train_data[train_data.iloc[:, 0].isin(range(4))].groupby(train_data.columns[0]).head(1000)
val_data = val_data[val_data.iloc[:, 0].isin(range(4))]
test_data = test_data[test_data.iloc[:, 0].isin(range(4))]
data_counts = val_data.groupby(val_data.columns[0]).size()
class_counts = train_data_1000['emotion'].value_counts()
print(class_counts)

# 計算訓練集的平均值和標準差
pixel_values = train_data_1000.iloc[:, 1].apply(lambda x: np.array(x.split(), dtype=np.uint8))
pixel_array = np.stack(pixel_values.values)
train_mean = np.mean(pixel_array) / 255.0
train_std = np.std(pixel_array) / 255.0

print(train_mean, train_std)

# 定義數據轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[train_mean] * 3, std=[train_std] * 3)  # 將均值和標準差擴展為3個通道
])

# 創建數據集對象
train_dataset = FER2013Dataset(train_data_1000, transform=transform)
val_dataset = FER2013Dataset(val_data, transform=transform)
test_dataset = FER2013Dataset(test_data, transform=transform)

# 創建數據加載器
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
