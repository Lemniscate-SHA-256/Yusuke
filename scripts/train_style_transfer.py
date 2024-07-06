import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from PIL import Image
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 4
learning_rate = 0.001
num_epochs = 2
style_weight = 1e6
content_weight = 1

# Directories
content_dir = 'data/content_images'
style_image_path = 'data/style_image.jpg'
output_dir = 'output/style_transfer'
os.makedirs(output_dir, exist_ok=True)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

# Dataset and DataLoader
dataset = datasets.ImageFolder(root=content_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the style image
style_image = Image.open(style_image_path).convert('RGB')
style_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])
style_image = style_transform(style_image).unsqueeze(0)

# Define the neural network
class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        
        self.relu = nn.ReLU()
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.upsample1(self.relu(self.conv4(x)))
        x = self.upsample2(self.relu(self.conv5(x)))
        x = self.conv6(x)
        return x

# Load the pre-trained VGG19 model
vgg = models.vgg19(pretrained=True).features
for param in vgg.parameters():
    param.requires_grad = False

# Extract content and style features
def get_features(image, model):
    layers = {'0': 'conv1_1',
              '5': 'conv2_1',
              '10': 'conv3_1',
              '19': 'conv4_1',
              '21': 'conv4_2',
              '28': 'conv5_1'}
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Calculate the gram matrix for style representation
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram 

# Initialize networks and optimizers
transformer = TransformerNet().cuda()
optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
mse_loss = nn.MSELoss()

# Style features
style_features = get_features(style_image.cuda(), vgg)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Training loop
for epoch in range(num_epochs):
    for i, (content_images, _) in enumerate(dataloader):
        content_images = content_images.cuda()

        # Generate transformed images
        transformed_images = transformer(content_images)
        
        # Extract features
        transformed_features = get_features(transformed_images, vgg)
        content_features = get_features(content_images, vgg)
        
        # Calculate content loss
        content_loss = content_weight * mse_loss(transformed_features['conv4_2'], content_features['conv4_2'])
        
        # Calculate style loss
        style_loss = 0
        for layer in style_grams:
            transformed_feature = transformed_features[layer]
            transformed_gram = gram_matrix(transformed_feature)
            _, d, h, w = transformed_feature.size()
            style_gram = style_grams[layer]
            style_loss += style_weight * mse_loss(transformed_gram, style_gram) / (d * h * w)
        
        # Total loss
        total_loss = content_loss + style_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {total_loss.item()}')

    # Save transformed images
    save_image(transformed_images.clamp(0, 255) / 255, os.path.join(output_dir, f'transformed_{epoch}.png'))

# Save model checkpoint
torch.save(transformer.state_dict(), 'models/style_transfer.pth')
