import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.utils.data as data
import torchvision.datasets as datasets

# Define an improved network model
class OptimizedNet(nn.Module):
    def __init__(self):
        super(OptimizedNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Calculate the Top-1 accuracy function
def calculate_top1_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    top1_accuracy = 100 * correct / total
    return top1_accuracy

# Main program
if __name__ == "__main__":
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the improved model
    model = OptimizedNet().to(device)

    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading Validation Sets (test_batch)
    test_dataset = datasets.CIFAR10(root='validation_dataset', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Define the optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Train the model
    # The following code is for example only, and the actual training requires a complete training process

    # Let's say we use pre-trained weights
    model.load_state_dict(torch.load('best_model2.pth', map_location=device, weights_only=True))

    # Calculate and output Top-1 accuracy
    top1_accuracy = calculate_top1_accuracy(model, test_loader, device)
    print(f"Top-1 Accuracy: {top1_accuracy:.18f}%")