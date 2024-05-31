from PIL import Image
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