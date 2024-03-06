from autoencoder import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from time import time
import numpy as np
import matplotlib.pyplot as plt

batch_size = 25

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
     transforms.RandomRotation(degrees=(-15, 15)),
     transforms.RandomHorizontalFlip()])

dataset = CIFAR10(root="./data/", train=True, download=True, transform=transform)

trainloader = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Autoencoder((3, 32, 32), 256).to(device)
model.load_state_dict(torch.load("./model/ae.pt"))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_img_grid(inputs):
    bs, _, _, _ = inputs.shape
    w = int(np.sqrt(bs))
    h = bs // w
    for i in range(w * h):
        plt.subplot(w, h, i + 1)
        plt.imshow(inputs[i].numpy().transpose((1, 2, 0)))
    plt.show()


loss_fn = nn.HuberLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 100
running_loss = 0.0
stime = time()
loss_trajectory = []

for epoch in range(num_epochs):
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data

        # show a grid of images for the first epoch and first batch
        if epoch == 0 and i == 0:
            show_img_grid(inputs)

        inputs = inputs.to(device)
        optimizer.zero_grad()

        preds = model(inputs)
        loss = loss_fn(preds, inputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print(f"[{epoch + 1} / {i}] Loss: {loss.item(): .3f} Time Taken: {time() - stime: .3f} secs")
            torch.save(model.state_dict(), f"./model/ae.pt")
            loss_trajectory.append(running_loss / 200)
            running_loss = 0.0
            stime = time()

plt.plot(loss_trajectory)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.show()
