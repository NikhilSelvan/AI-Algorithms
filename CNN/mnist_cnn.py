# Import required libraries
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms
import torchvision  # Computer vision utilities
import torchvision.transforms as transforms  # Image transformations
from torch.utils.data import DataLoader  # Data loading utilities
import matplotlib.pyplot as plt  # For visualization
import numpy as np  # For numerical operations
from collections import defaultdict

# Set random seed for reproducibility
# This ensures that the results are consistent across different runs
torch.manual_seed(42)

# Check if CUDA (GPU) is available and set the device accordingly
# This allows the model to run on GPU if available, which significantly speeds up training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels
        # kernel_size=3 means 3x3 filter, padding=1 maintains spatial dimensions
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # Second convolutional layer: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling layer: reduces spatial dimensions by half
        self.pool = nn.MaxPool2d(2, 2)
        
        # First fully connected layer: 64*7*7 input features (after pooling), 128 output features
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # Output layer: 128 input features, 10 output features (one for each digit 0-9)
        self.fc2 = nn.Linear(128, 10)
        
        # ReLU activation function: introduces non-linearity
        self.relu = nn.ReLU()
        
        # Dropout layer: randomly sets 25% of inputs to zero during training to prevent overfitting
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # First conv block: conv -> relu -> pool
        x = self.pool(self.relu(self.conv1(x)))
        
        # Second conv block: conv -> relu -> pool
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # First fully connected layer with ReLU activation
        x = self.relu(self.fc1(x))
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Final output layer
        x = self.fc2(x)
        return x

# Define data transformations
transform = transforms.Compose([
    # Convert PIL image to tensor and scale pixel values to [0,1]
    transforms.ToTensor(),
    # Normalize the data using MNIST dataset mean and std
    # This helps in faster and more stable training
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load training dataset
# download=True will download the dataset if it's not already present
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                         train=True,
                                         transform=transform,
                                         download=True)

# Load test dataset
test_dataset = torchvision.datasets.MNIST(root='./data',
                                        train=False,
                                        transform=transform)

# Create data loaders
# batch_size=64 means we process 64 images at a time
# shuffle=True randomizes the order of training data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the model and move it to the appropriate device (CPU/GPU)
model = CNN().to(device)

# Define loss function (Cross Entropy Loss for classification)
criterion = nn.CrossEntropyLoss()

# Initialize optimizer (Adam with learning rate 0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add before the train function
def plot_learning_curves(train_losses, train_accs, test_accs):
    """
    Plot learning curves showing training loss and accuracy over epochs
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot training loss
    ax1.plot(train_losses, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Epochs')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(test_accs, label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Model Accuracy over Epochs')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Modify the train function
def train(epochs):
    model.train()
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_train_acc = 100 * correct / total
        epoch_test_acc = evaluate()
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(epoch_test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}] - '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_train_acc:.2f}%, '
              f'Test Acc: {epoch_test_acc:.2f}%')
    
    # Plot learning curves
    plot_learning_curves(train_losses, train_accs, test_accs)
    return train_losses, train_accs, test_accs

# Modify the evaluate function to return accuracy without printing
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# Function to visualize model predictions
def visualize_predictions():
    model.eval()
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images = images[:5].to(device)  # Get first 5 images
    labels = labels[:5]
    
    # Get model predictions
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # Create visualization
    fig = plt.figure(figsize=(12, 4))
    for i in range(5):
        ax = fig.add_subplot(1, 5, i + 1)
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f'Pred: {predicted[i]}\nTrue: {labels[i]}')
        ax.axis('off')
    plt.show()

# Main execution
print("Starting training...")
train_losses, train_accs, test_accs = train(epochs=5)  # You can change the number of epochs here

print("\nVisualizing predictions...")
visualize_predictions()

# Save the model
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("\nModel saved as 'mnist_cnn.pth'") 