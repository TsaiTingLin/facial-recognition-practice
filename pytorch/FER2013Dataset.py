import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class FER2013Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 提取图像像素和标签
        image = np.fromstring(self.dataframe.iloc[idx, 1], dtype=int, sep=' ')
        image = image.reshape(48, 48).astype(np.uint8)
        image = Image.fromarray(image)  # 将NumPy数组转换为PIL图像
        image = image.convert('RGB')  # 将灰度图像转换为RGB图像

        if self.transform:
            image = self.transform(image)

        label = int(self.dataframe.iloc[idx, 0])
        return image, label


# 定义情绪标签字典
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小以匹配MobileNetV2的输入大小
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 使用3通道的均值和标准差进行归一化
])

# 读入数据集
df_raw = pd.read_csv('../fer2013.csv')

# 按Usage栏位划分数据
train_data = df_raw[df_raw['Usage'] == 'Training']
val_data = df_raw[df_raw['Usage'] == 'PublicTest']
test_data = df_raw[df_raw['Usage'] == 'PrivateTest']

# 创建数据集对象
train_dataset = FER2013Dataset(train_data, transform=transform)
val_dataset = FER2013Dataset(val_data, transform=transform)
test_dataset = FER2013Dataset(test_data, transform=transform)

# 创建数据加载器
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")
