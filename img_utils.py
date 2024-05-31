from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
import cv2

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

def get_diff(img1, img2):
    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    img_diff = np.abs(img1 - img2)
    return img_diff.astype(np.uint8)


def save_img(path, img):
    if img.dtype == np.uint8:
        cv2.imwrite(path, img)
    elif img.dtype == np.float32 or img.dtype == np.float16 or img.dtype == np.float64:
        cv2.imwrite(path, img.astype(np.uint8) if img.max() > 1.0 else  (img * 255).astype(np.uint8))