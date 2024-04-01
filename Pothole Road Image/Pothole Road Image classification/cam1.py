import torch
from torchvision import models, transforms, utils
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained model
base_model = models.resnet50(pretrained=True)

# Select an intermediate layer
layer_name = 'layer2'  # You can choose the layer you want
num_layers = 5  # You can adjust this number based on the layer you choose
intermediate_layer_model = torch.nn.Sequential(*list(base_model.children())[:num_layers])

# Preprocess image
img_path = "D:\\桌面\\datasets\\train\\normal\\normal1.jpg"  # Your image path
img = Image.open(img_path)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img_t = preprocess(img)
img_u = torch.unsqueeze(img_t, 0)

# Predict image
base_model.eval()
with torch.no_grad():
    intermediate_output = intermediate_layer_model(img_u)

# Select a subset of the feature maps
num_feature_maps = 8  # You can adjust this number
intermediate_output = intermediate_output[:, :num_feature_maps, :, :]

# Create a grid for displaying the feature maps
grid = utils.make_grid(intermediate_output[0], nrow=num_feature_maps, normalize=True)

# Display each feature map separately
plt.figure(figsize=(12, 12))
for i in range(num_feature_maps):
    plt.subplot(1, num_feature_maps, i + 1)
    plt.imshow(grid[i].cpu().numpy(), cmap='viridis')  # You can choose a different colormap if needed
    plt.axis('off')

plt.show()