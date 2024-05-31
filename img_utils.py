from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F

def read_tensor_img(path: str) -> torch.Tensor:
    """Read an image with PIL.Image and convert it to a Tensor

    Args:
        path (str): path to the image 
        
    Returns:
        A tensor of image 
        shape: [1, c, h, w]
        value: [0.0, 1.0]
    """
    # read image
    img = Image.open(path).convert('RGB')
    
    return F.to_tensor(img).unsqueeze(0)


def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy