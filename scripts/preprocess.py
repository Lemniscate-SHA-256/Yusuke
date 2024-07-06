import os
from PIL import Image
import torchvision.transforms as transforms

data_dir = 'data/manga_images'
output_dir = 'data/processed_images'

os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

for img_name in os.listdir(data_dir):
    img_path = os.path.join(data_dir, img_name)
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img)
    img_tensor.save(os.path.join(output_dir, img_name))
