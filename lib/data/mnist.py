import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class MNIST(Dataset):
    def __init__(self, root, mode):
        train = (mode == 'train')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.mnist = torchvision.datasets.MNIST(root, train=train, transform=transform, download=False)
        
    def __getitem__(self, index):
        """
        :return: (1, 28, 28), Float, binary
        """
        return self.mnist[index][0]
        
    def __len__(self):
        return len(self.mnist)
        
        
if __name__ == '__main__':
    mnist = MNIST('data/MNIST', 'train')
    # print(mnist[0].max())
    img = mnist[10].numpy()[0]
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap='gray')
    plt.show()
