import os
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Define the image transformations
transform = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(contrast=(0.6, 1.4), brightness=(0.6, 1.4)),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Specify the path to your dataset
data_dir = 'pre_cropped_training_pics/'

# Create a dataset using ImageFolder
dataset = ImageFolder(root=data_dir, transform=transform)

# Create a DataLoader for batch processing
batch_size = 1  # Set batch size to the number of images to generate together
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Specify the path to save augmented images
save_path = 'new_data/'

# Iterate through the data loader and save augmented images
for i, (images, labels) in enumerate(data_loader):
    # Generate augmented images for each image in the batch
    augmented_images = []

    # Convert tensor to PIL Image and ensure RGB mode
    image = transforms.ToPILImage()(images[0].cpu())

    for j in range(5):
        # Apply augmentations and save the image
        augmented_image = transform(image)
        augmented_image = (transforms.ToPILImage()(augmented_image.cpu()))
        class_folder = os.path.join(save_path, dataset.classes[labels[0]])
        os.makedirs(class_folder, exist_ok=True)
        image_path = os.path.join(class_folder, f'image_{i}_{j}.jpeg')
        augmented_image.save(image_path)
        print(f'Saved: {image_path}')

    # Display the augmented images in a grid
    grid_image = utils.make_grid(images, nrow=batch_size, normalize=True)
    plt.imshow(grid_image.permute(1, 2, 0))  # Change the order of dimensions for matplotlib
    plt.title(f'Class: {dataset.classes[labels[0]]}')
    plt.pause(0.001)  # Pause for 1 millisecond
    plt.clf()  # Clear the current figure

print('Image augmentation and saving complete.')
