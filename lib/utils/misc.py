import torch
import numpy as np
import cv2


def tonumpyimg(img):
    """
    Convert a normalized tensor image to unnormalized uint8 numpy image
    For single channel image, no unnormalization is done.

    :param img: torch, normalized, (3, H, W), (H, W)
    :return: numpy: (H, W, 3), (H, W). uint8
    """
    
    return touint8(tonumpy(unnormalize_torch(img)))


def tonumpy(img):
    """
    Convert torch image map to numpy image map
    Note the range is not change

    :param img: tensor, shape (C, H, W), (H, W)
    :return: numpy, shape (H, W, C), (H, W)
    """
    if len(img.size()) == 2:
        return img.cpu().detach().numpy()
    
    return img.permute(1, 2, 0).cpu().detach().numpy()


def touint8(img):
    """
    Convert float numpy image to uint8 image
    :param img: numpy image, float, (0, 1)
    :return: uint8 image
    """
    img = img * 255
    return img.astype(np.uint8)


def normalize_torch(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Normalize a torch image.
    :param img: (3, H, W), in range (0, 1)
    """
    img = img.clone()
    img -= torch.tensor(mean).view(3, 1, 1)
    img /= torch.tensor(std).view(3, 1, 1)
    
    return img


def unnormalize_torch(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized Tensor image to unnormalized form
    For single channel image, no normalization is done.
    :param img: (C, H, W), (H, W)
    """
    if img.size()[0] == 3:
        img = img.clone()
        img *= torch.Tensor(std).view(3, 1, 1)
        img += torch.Tensor(mean).view(3, 1, 1)
    
    return img


def gray2RGB(img_raw):
    """
    Convert a gray image to RGB
    :param img_raw: (H, W, 3) or (H, W), uint8, numpy
    :return: (H, W, 3)
    """
    if len(img_raw.shape) == 2:
        img_raw = np.repeat(img_raw[:, :, None], 3, axis=2)
    if img_raw.shape[2] > 3:
        img_raw = img_raw[:, :, :3]
    return img_raw


def color_scale(attention):
    """
    Visualize a attention map
    :param scale_map: (C, H, W), attention map, softmaxed
    :return: (3, H, W), colored version
    """
    
    colors = torch.Tensor([
        [1, 0, 0],  # red
        [0, 1, 0],  # green
        [0, 0, 1],  # blue
        [0, 0, 0],  # black
    ]).float()
    
    # (H, W)
    attention = torch.argmax(attention, dim=0)
    # (H, W, C)
    color_map = colors[attention]
    color_map = color_map.permute(2, 0, 1)
    
    return color_map


def warp_torch(map, H):
    """
    Warp a torch image.
    :param map: either (C, H, W) or (H, W)
    :param H: (3, 3)
    :return: warped iamge, (C, H, W) or (H, W)
    """
    map = tonumpy(map)
    h, w = map.shape[-2:]
    map = cv2.warpPerspective(map, H, dsize=(w, h))
    
    return totensor(map)


def torange(array, low, high):
    """
    Render an array to value range (low, high)
    :param array: any array
    :param low, high: the range
    :return: new array
    """
    min, max = array.min(), array.max()
    # normalized to [0, 1]
    array = array - min
    array = array / (max - min)
    # to (low, high)
    array = array * (high - low) + low
    
    return array


def tofloat(img):
    """
    Convert a uint8 image to float image
    :param img: numpy image, uint8
    :return: float image
    """
    return img.astype(np.float) / 255


def tonumpy_batch(imgs):
    """
    Convert a batch of torch images to numpy image map

    :param imgs: (B, C, H, W)
    :return: (B, H, W, C)
    """
    
    return imgs.permute(0, 2, 3, 1).cpu().detach().numpy()


def totensor(img, device=torch.device('cpu')):
    """
    Do the reverse of tonumpy
    """
    if len(img.shape) == 2:
        return torch.from_numpy(img).to(device).float()
    return torch.from_numpy(img).permute(2, 0, 1).to(device).float()


def totensor_batch(imgs, device=torch.device('cpu')):
    """
    Do the reverse of tonumpy_batch
    """
    return torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device).float()


def RGB2BGR(*imgs):
    return [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in imgs]


def unnormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a normalized tensor image to unnormalized form
    :param img: (B, C, H, W)
    """
    img = img.detach().cpu()
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    
    return img


def toUint8RGB(img):
    return (tonumpy(unnormalize(img)) * 255.).astype(np.uint8)

