import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.utils.data as data
import torchvision.datasets as datasets

# Define the network model (consistent with the model structure at the time of training)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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

    # Load the trained model
    model = Net().to(device)
    # Load the model weights and set weights_only=True
    model.load_state_dict(torch.load('best_model2.pth', map_location=device, weights_only=True))

    # Define image preprocessing
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize image to 32x32 (same as CIFAR-10 input)
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Loading Validation Sets (test_batch)
    # test_batch files are located in the 'validation_dataset' directory
    test_dataset = datasets.CIFAR10(root='validation_dataset', train=False, download=False, transform=transform)
    # Set download=True
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Calculate and output Top-1 accuracy
    top1_accuracy = calculate_top1_accuracy(model, test_loader, device)
    print(f"Top-1 Accuracy: {top1_accuracy:.18f}%")