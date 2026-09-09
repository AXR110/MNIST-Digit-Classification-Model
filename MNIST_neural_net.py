import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import ssl

# Bypass Mac SSL verification
ssl._create_default_https_context = ssl._create_unverified_context

# Load and normalize the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

# Define a simple Feed-Forward Neural Network
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 128),            # 28x28 pixel input
            nn.ReLU(),                                      # Activation function
            nn.Linear(128, 10)       # 10 output classes (digits 0-9)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.network(x)

model = SimpleNeuralNet()

# Define the loss function and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 3
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()               # Clear old gradients
        outputs = model(images)             # Forward pass
        loss = criterion(outputs, labels)   # Calculate error
        loss.backward()                     # Backward pass (compute gradients)
        optimizer.step()                    # Update weights

    print(f"Epoch {epoch+1}/{epochs} complete. Training Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Should reveal 90%+ accuracy
print(f"Accuracy on 10,000 test images: {100 * correct / total:.2f}%")

#
# **Visualization**
#
import matplotlib.pyplot as plt
import numpy as np

# Grab a single batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Pass the images through the network to get predictions
model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# Create a 2x5 grid to show the first 10 images
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flatten()):
    # The images were normalized during loading, so they have to be un-normalized to be viewed properly
    img = images[i].numpy().squeeze()
    img = (img * 0.5) + 0.5

    # Display the image in grayscale
    ax.imshow(img, cmap='gray')
    ax.axis('off')

    # Label the image number guess alongside the real value: Green if correct, Red if the network guessed wrong
    color = 'green' if predicted[i] == labels[i] else 'red'
    ax.set_title(f"Predicted: {predicted[i].item()}\nTrue: {labels[i].item()}", color=color)

plt.tight_layout()
plt.show()  # Opens the window
