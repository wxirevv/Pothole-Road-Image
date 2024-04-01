
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Creating the result directory
result_dir = "D:\\桌面\\datasets\\results_efficientnet"
os.makedirs(result_dir, exist_ok=True)

# Defining the dataset class
class RoadDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

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

# Define data augmentation operations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
data_dir = "D:\\桌面\\data\\dataaugmentation"
dataset = RoadDataset(data_dir, transform=transform)

# Normalize the dataset
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
normalized_dataset = []
for i in range(len(dataset)):
    image, _ = dataset[i]
    image = (image - mean[:, None, None]) / std[:, None, None]
    normalized_dataset.append((image, _))

# Plot the pixel density histogram
pixel_density = []
for image, _ in normalized_dataset:
    pixel_density.append(image.mean().item())  # Use item() to get the Python scalar value

plt.hist(pixel_density, bins=50, alpha=0.5)
plt.xlabel('Pixel Density')
plt.ylabel('Frequency')
plt.title('Pixel Density Histogram')
plt.show()
