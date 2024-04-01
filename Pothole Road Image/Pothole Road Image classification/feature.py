import filecmp
import os
import cv2
import numpy as np
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import time
import pandas as pd
import matplotlib.pyplot as plt

# 创建结果目录
result_dir = "D:\\桌面\\datasets\\results_efficientnet"
os.makedirs(result_dir, exist_ok=True)

# 定义数据集类
class RoadDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)
        self.features = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if 'normal' not in img_name else 0

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_features(self, model):
        self.features = []
        model.eval()
        with torch.no_grad():
            for images, _ in self:
                images = images.unsqueeze(0).to(device)
                features = model(images)
                self.features.append(features.squeeze().cpu().numpy())

# 定义数据增强操作
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据集
data_dir = "D:\\桌面\\data\\dataaugmentation"
dataset = RoadDataset(data_dir, transform=transform)

# 创建EfficientNet模型实例
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Identity()  # 移除全连接层

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 从数据集中提取特征
dataset.get_features(model)

# 创建一个DataFrame以保存特征
df = pd.DataFrame(dataset.features, columns=[f"Feature_{i}" for i in range(len(dataset.features[0]))])

# 保存特征到Excel文档
features_file = os.path.join(result_dir, "D:\\桌面\\datasets\\feature1.xlsx")
df.to_excel(features_file, index=False)

# 寻找坑洼图像的索引
bumpy_image_index = None
for i, img_name in enumerate(dataset.image_files):
    if 'potholes' in img_name:
        bumpy_image_index = i
        break

if bumpy_image_index is not None:
    # 获取坑洼图像的特征
    bumpy_features = df.loc[bumpy_image_index, :].values

    # 可视化坑洼图像的特征
    plt.plot(bumpy_features)
    plt.xlabel("Feature Index")
    plt.ylabel("Feature Value")
    plt.title("Bumpy Image Features")
    plt.show()

    # 显示坑洼图像
    bumpy_image = cv2.imread(os.path.join(data_dir, dataset.image_files[bumpy_image_index]))
    bumpy_image = cv2.cvtColor(bumpy_image, cv2.COLOR_BGR2RGB)
    plt.imshow(bumpy_image)
    plt.axis('off')
    plt.title("Bumpy Image")
    plt.show()
else:
    print("没有找到坑洼图像。")

# 可视化特征分布情况
plt.figure(figsize=(8, 6))
plt.hist(df.values.flatten(), bins=50, color='skyblue', alpha=0.7)
plt.xlabel("Feature Value")
plt.ylabel("Frequency")
plt.title("Feature Distribution")
plt.show()
