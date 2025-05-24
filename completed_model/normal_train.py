import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import warnings
from tqdm import tqdm

# Ignore warning messages
warnings.filterwarnings("ignore")

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize according to the dataset
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(10),  # Random rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use ImageNet mean and standard deviation
])

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Iterate through all image files and use the file names as categories
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

# Define the VGG16 model with added attention mechanisms
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
        # Obtain the classification part of VGG16
        self.classifier = vgg.classifier
        # Modify the classification layer to match the number of classes in the dataset
        self.classifier[6] = nn.Linear(self.classifier[6].in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.ca(x)
        x = self.sa(x)
        x = self.avgpool(x)
        # Flatten the feature maps.
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the training and validation sets
train_dataset = CustomDataset(root_dir=r'.\datasets\Normal\facescape_dataset\facescape_128', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Validation set
val_dataset = CustomDataset(root_dir=r'.\datasets\Normal\facescape_dataset\facescape_128_validation', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the VGG16 model with added attention mechanisms
num_classes = len(train_dataset.class_to_idx)
model = VGG16WithAttention(num_classes=num_classes)

# Transfer the model to the GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Learning rate

# Train the model
num_epochs = 100  # Increase the number of training epochs.
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    # Use tqdm to display the progress bar.
    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        pbar.set_postfix({'Loss': loss.item()})  # Update the progress bar information

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Normal epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    # Evaluation on the validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation', unit='batch'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    error_rate = 100 * (total - correct) / total  # Error rate as EER
    print(f'Validation - Epoch {epoch + 1}, ACC: {accuracy:.6f}%, EER: {error_rate:.6f}%')

# Save the model weights
os.makedirs(r'.\saved_models', exist_ok=True)
torch.save(model.state_dict(), r'.\saved_models\normal_facescape_model.pth')