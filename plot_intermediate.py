import plotly.graph_objs as go
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from autoencoder import Autoencoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


batch_size = 25
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def my_normalization(x):
    return x

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(my_normalization)])

dataset = CIFAR10(root="./data/", train=True, download=True, transform=transform)

trainloader = DataLoader(dataset, batch_size=batch_size,
                         shuffle=True)

network = Autoencoder((3, 32, 32), nhid=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network.to(device)
save_name = "model/ae.pt"

network.load_state_dict(torch.load(save_name))

network.eval()
labels = []
representations = []

for batch in tqdm(trainloader):
    imgs, lbls = batch
    labels += lbls

    internal_rep = network.get_representation(imgs.to(device))
    representations += internal_rep.tolist()

representations = np.float32(representations)

# pca = PCA(n_components=3)  
# X_pca = pca.fit_transform(representations)
tsne = TSNE(n_components=3, perplexity=30.0, n_iter=300, verbose=1)
X_pca = tsne.fit_transform(representations)

# Plot the reduced data in 3D
fig = go.Figure()

# Unique class labels
unique_labels = np.unique(labels)

# Color map for classes
colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'grey', 'black']

for i, label in tqdm(enumerate(unique_labels)):
    # Select indices for data points with the current class label
    idx = np.where(labels == label)
    
    # Add trace for each class
    fig.add_trace(go.Scatter3d(
        x = X_pca[idx, 0].flatten(), 
        y = X_pca[idx, 1].flatten(), 
        z = X_pca[idx, 2].flatten(), 
        mode = 'markers',
        marker = dict(
            size = 3,
            color = colors[i % len(colors)],  # Cycle through colors
        ),
        name = f'{classes[label]}'
    ))

# Set chart and axes titles
fig.update_layout(
    title = "3D PCA Visualization of the Dataset",
    scene = dict(
        xaxis_title='PCA 1',
        yaxis_title='PCA 2',
        zaxis_title='PCA 3'
    )
)

# Show the plot
fig.show()
