import os
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset
from PIL import Image
import warnings
import json


warnings.filterwarnings("ignore")

# Test data preprocessing (without data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class (used to load the training set to get category information)
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Iterate through all image files and use the file name as the category
        for filename in os.listdir(root_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                class_name = filename  # Use the file name as the category
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

# Load the training set to get the category information
train_dataset = CustomDataset(root_dir=r'.\datasets\RGB\facescape_dataset\facescape_128', transform=test_transform)

# Initialize the ResNet50 model and modify the fully connected layer to fit the number of categories in the dataset
model = models.resnet50(pretrained=True)  # Use pre-training weights
num_classes = len(train_dataset.class_to_idx)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Transfer the model to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loads the saved model weights
model.load_state_dict(torch.load(r'.\saved_models\rgb_facescape_model.pth'))
model.eval()

# Define a function to predict a single image
def predict_single_image(image_path, idx_to_class):
    image = Image.open(image_path).convert('RGB')
    image = test_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        topk_prob, topk_catid = torch.topk(probabilities, num_classes)  # Use the total number of classifications

    # Get the category name
    results = {}
    for i in range(num_classes):
        class_name = idx_to_class[topk_catid[0][i].item()].split('.')[0]
        results[class_name] = topk_prob[0][i].item()

    return results

def predict_and_save_rgb():
    test_image = r".\single_test_image\test.jpg"
    results = predict_single_image(test_image, train_dataset.idx_to_class)
    return results