import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from efficientnet_pytorch import EfficientNet
import torch.nn as nn

# Define the test dataset class
class TestRoadDataset(Dataset):
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

        if self.transform:
            image = self.transform(image)

        return image, img_name

# Define the model class
class RoadClassifier(nn.Module):
    def __init__(self, num_classes):
        super(RoadClassifier, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Identity()  # Remove the fully connected layer to add a custom classification layer
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.efficientnet(x)
        output = self.classifier(features)
        return output

# Define data augmentation operations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the test dataset
test_data_dir = "D:\\Datasets\\testdata_V2"
test_dataset = TestRoadDataset(test_data_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the trained model
model_path = "C:\\Users\\19666\\Desktop\\fold_5_epoch_10.pt"  # Replace with the path of your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RoadClassifier(num_classes=1)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Classify the test images
results = []
with torch.no_grad():
    for images, filenames in test_dataloader:
        images = images.to(device)
        outputs = model(images)
        predicted_labels = torch.round(outputs).cpu().numpy()
        for filename, label in zip(filenames, predicted_labels):
            results.append((filename, int(label)))

# Save the results to a CSV file
results_df = pd.DataFrame(results, columns=["filename", "predicted_label"])
results_df.to_csv("test_results.csv", index=False)
