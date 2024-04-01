import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import time

# Creating the result directory
result_dir = "D:\\桌面\\datasets\\results_resnet50"
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
        label = 0 if 'normal' not in img_name else 1

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
data_dir = "D:\\桌面\\data\\dataaugmentation\\dataaugmentation"
dataset = RoadDataset(data_dir, transform=transform)

# Oversample the minority class
normal_indices = [idx for idx, (_, label) in enumerate(dataset) if label == 0]
bumpy_indices = [idx for idx, (_, label) in enumerate(dataset) if label == 1]
bumpy_indices_resampled = resample(bumpy_indices, replace=True, n_samples=len(normal_indices))

# Merge the sample indices
balanced_indices = normal_indices + bumpy_indices_resampled

# Create the balanced dataset instance
balanced_dataset = torch.utils.data.Subset(dataset, balanced_indices)
batch_size = 32
dataloader = DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)

# Define the model class
class RoadClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RoadClassifier, self).__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer to add a custom classification layer
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.resnet(x)
        output = self.classifier(features)
        return output

# Define evaluation metrics
def evaluate(model, dataloader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted_labels = torch.round(outputs)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    return accuracy, recall, f1, auc

# Use normal train-validation split
train_size = int(0.8 * len(balanced_dataset))
val_size = len(balanced_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(balanced_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train and save the model for each epoch
num_epochs = 50

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model = RoadClassifier(num_classes=1)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    running_loss = 0.0
    start_time = time.time()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).float())

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    end_time = time.time()
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Loss: {epoch_loss:.4f}, Training Time: {end_time - start_time:.2f} seconds")

    # Evaluate the model on the validation set
    start_time = time.time()
    accuracy, recall, f1, auc = evaluate(model, val_loader)
    end_time = time.time()
    print(f"Validation Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, Inference Time: {end_time - start_time:.2f} seconds")

    # Save the model
    model_path = os.path.join(result_dir, f"model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), model_path)