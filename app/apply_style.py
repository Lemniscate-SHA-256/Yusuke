from .style_transfer import StyleTransferModel
import torch
from PIL import Image

style_model = StyleTransferModel('path_to_style_image.jpg').cuda()
style_model.load_state_dict(torch.load('models/style_transfer.pth'))

def apply_style(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).cuda()
    styled_image = style_model(image_tensor)
    styled_image_path = 'styled_' + image_path
    save_image(styled_image, styled_image_path)
    return styled_image_path
