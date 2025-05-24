import os
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset
from PIL import Image
import warnings
import cv2
import numpy as np
import json

# Ignore warning messages
warnings.filterwarnings("ignore")

# Test data preprocessing (without data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.486], std=[0.229, 0.224, 0.225])
])

# Custom dataset class (for loading the training set to obtain class information)
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Iterate through all image files and use the file names as categories.
        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                class_name = filename  # Use the file names as categories
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                    self.idx_to_class[len(self.idx_to_class)] = class_name
                self.image_paths.append(os.path.join(root_dir, filename))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the channel attention module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * (avg_out + max_out)

# Define the spatial attention module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y

# Define the VGG16 model with added attention mechanisms (identical to the training code)
class VGG16WithAttention(nn.Module):
    def __init__(self, num_classes):
        super(VGG16WithAttention, self).__init__()
        # Load the pre-trained VGG16 model
        vgg = models.vgg16(pretrained=True)
        # Obtain the feature extraction part of VGG16
        self.features = vgg.features
        # Add attention mechanisms
        self.ca = ChannelAttention(512)
        self.sa = SpatialAttention()
        # Add a global average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # Use the original classifier structure
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),  # Input dimension 512*7*7
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.ca(x)  # Channel Attention
        x = self.sa(x)  # Spatial Attention
        x = self.avgpool(x)  # Global Average Pooling
        x = torch.flatten(x, 1)  # Flatten the features
        x = self.classifier(x)
        return x

# Load the training set to obtain class information
train_dataset = CustomDataset(root_dir=r'.\datasets\Normal\facescape_dataset\facescape_128', transform=test_transform)

# Initialize the VGG16 model with added attention mechanisms
num_classes = len(train_dataset.class_to_idx)
model = VGG16WithAttention(num_classes=num_classes)

# Transfer the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load the saved model weights
model.load_state_dict(torch.load(r'.\saved_models\normal_facescape_model.pth'))
model.eval()

# Define a function to predict a single image
def predict_single_image(image_path, idx_to_class):
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)  # Convert the PIL image to a numpy array

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Calculate the normal map
    height_map = gray_image.astype(np.float32) / 255.0
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    gradient_x = cv2.filter2D(height_map, -1, kernel_x)
    gradient_y = cv2.filter2D(height_map, -1, kernel_y)

    normal_x = gradient_x * 20.0  # scale=20.0
    normal_y = gradient_y * 20.0
    normal_z = np.sqrt(1 - np.clip(normal_x**2 + normal_y**2, 0, 1))

    normal_x = normal_x / np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_y = normal_y / np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_z = normal_z / np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)

    normal_map = np.stack((normal_x + 1, normal_y + 1, normal_z + 1), axis=2) / 2.0
    normal_map = (normal_map * 255).astype(np.uint8)

    # Convert the normal map to a PIL image and preprocess it
    normal_map_pil = Image.fromarray(normal_map)
    normal_map_pil = test_transform(normal_map_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(normal_map_pil)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        topk_prob, topk_catid = torch.topk(probabilities, num_classes)  # Use the total number of classes

    # Get the class name
    results = {}
    for i in range(num_classes):
        class_name = idx_to_class[topk_catid[0][i].item()].split('.')[0]  # Remove the file extension
        results[class_name] = topk_prob[0][i].item()

    return results

def predict_and_save_normal():
    test_image = r".\single_test_image\test.jpg"
    results = predict_single_image(test_image, train_dataset.idx_to_class)
    return results