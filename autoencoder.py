import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, shape, nhid=32) -> None:
        super(Encoder, self).__init__()
        c, w, h = shape

        # (c, 32, 32) -> (16, 32, 32)
        self.conv1 = nn.Conv2d(c, 16, 3, padding="same")

        # (16, 16, 16) -> (32, 16, 16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding="same")

        # (32, 8, 8) -> (64, 8, 8)
        self.conv3 = nn.Conv2d(32, 64, 3, padding="same")

        # (64, 4, 4) -> (128, 4, 4)
        self.conv4 = nn.Conv2d(64, 128, 3, padding="same")
        
        self.maxpool = nn.MaxPool2d(2, 2)
        self.FC = nn.Linear(128 * 4 * 4, nhid)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x = self.maxpool(x)
        
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x = nn.Flatten()(x)

        x = self.FC(x)
        return x

class Decoder(nn.Module):
    def __init__(self, shape, nhid=32) -> None:
        super(Decoder, self).__init__()

        c, w, h = shape

        self.FC = nn.Linear(nhid, 128 * 4 * 4)

        # (128, 4, 4) -> (64, 8, 8)
        self.conv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)

        # (64, 8, 8) -> (32, 16, 16)
        self.conv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

        # (32, 16, 16) -> (c, 32, 32)
        self.conv3 = nn.ConvTranspose2d(32, c, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.FC(x)

        x = x.view(-1, 128, 4, 4)

        x = self.conv1(x)
        x = nn.ReLU()(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        
        x = self.conv3(x)
        
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, shape, nhid=32) -> None:
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(shape, nhid)
        self.decoder = Decoder(shape, nhid)

    def forward(self, x):
        x = self.encoder(x)

        x = nn.ReLU()(x)

        x = self.decoder(x)

        return nn.Sigmoid()(x)
            
    def get_representation(self, x):
        x = self.encoder(x)
        return x
