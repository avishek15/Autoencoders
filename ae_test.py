import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from time import time
from autoencoder import Autoencoder

batch_size = 25

def my_normalization(x):
    return x

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(my_normalization)])

dataset = CIFAR10(root="./data/", train=True, download=True, transform=transform)

trainloader = iter(DataLoader(dataset, batch_size=batch_size,
                         shuffle=True))

@torch.no_grad()
def generate_images():
    network = Autoencoder((3, 32, 32), nhid=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    save_name = "model/ae.pt"

    network.load_state_dict(torch.load(save_name))

    img_batch, _ = next(trainloader)

    images = network(img_batch.to(device))
    np_images = images.cpu().detach().numpy().transpose((0, 2, 3, 1))

    for i in range(len(np_images)):
        plt.subplot(5, 5, i + 1)
        plt.imshow(np_images[i])
    plt.show()


if __name__ == '__main__':
    generate_images()
