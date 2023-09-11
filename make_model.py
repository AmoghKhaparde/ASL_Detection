import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N' ,'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
num_classes = len(classes)

# Define data transformations with data augmentation, including color changes
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ASL training dataset
train_dataset = datasets.ImageFolder(root='new_training_images/', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load ASL test dataset (you need to provide the path to your test data)
test_dataset = datasets.ImageFolder(root='test_images/', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)  # No need to shuffle for testing

# Load pre-trained ResNet-18 model
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
num_features = model.fc.in_features

# Modify classifier for ASL characters
model.fc = nn.Linear(num_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15

# Lists to store training and testing loss and accuracy for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct_predictions / total_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # # Evaluate the model on the test dataset
    # model.eval()
    # test_loss = 0
    # correct_predictions = 0
    # total_samples = 0

    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)

    #         test_loss += loss.item()

    #         _, predicted = torch.max(outputs, 1)
    #         correct_predictions += (predicted == labels).sum().item()
    #         total_samples += labels.size(0)

    # test_loss /= len(test_loader)
    # test_accuracy = correct_predictions / total_samples
    # test_losses.append(test_loss)
    # test_accuracies.append(test_accuracy)

    # print(f"Epoch [{epoch+1}/{num_epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# # Plot loss and accuracy curves
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(test_losses, label='Test Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Loss Curves')

# plt.subplot(1, 2, 2)
# plt.plot(train_accuracies, label='Train Accuracy')
# plt.plot(test_accuracies, label='Test Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.title('Accuracy Curves')

# plt.tight_layout()
# plt.show()

# Save the model's state_dict
torch.save(model.state_dict(), 'data/asl_model19.pth')