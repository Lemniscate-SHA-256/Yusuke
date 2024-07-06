import torch
from torchvision.utils import save_image
from .gan_model import Generator

generator = Generator().cuda()
generator.load_state_dict(torch.load('models/generator.pth'))

def generate_panel():
    z = torch.randn(1, 100, 1, 1).cuda()
    generated_image = generator(z)
    save_image(generated_image, 'generated.png')
    return 'generated.png'
