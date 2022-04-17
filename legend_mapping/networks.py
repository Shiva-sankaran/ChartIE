import torch
from torch import nn

class Autoencoder_v1(nn.Module):
    def __init__(self):
        super(Autoencoder_v1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=20*40*3, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=20*40*3),
            nn.BatchNorm1d(20*40*3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 3, 20, 40)
        return x

class Autoencoder_v2(nn.Module):
    def __init__(self):
        super(Autoencoder_v2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=20*40*3, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=20*40*3),
            nn.BatchNorm1d(20*40*3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten image
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0), 3, 20, 40)
        return x

class Conv_Autoencoder_v1(nn.Module):
    def __init__(self):
        super(Conv_Autoencoder_v1, self).__init__()
        self.encoder = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential (
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Conv_Autoencoder_v2(nn.Module):
    def __init__(self):
        super(Conv_Autoencoder_v2, self).__init__()
        self.encoder = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential (
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




class Encoder_v2(nn.Module):
    def __init__(self):
        super(Encoder_v2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=20*40*3, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=20*40*3),
            nn.BatchNorm1d(20*40*3),
            nn.ReLU(),
        )
    def forward(self, x):
        x = x.view(x.size(0), -1) # Flatten image
        x = self.encoder(x)
        return x


class Decoder_v2(nn.Module):
    def __init__(self):
        super(Decoder_v2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=20*40*3, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=20*40*3),
            nn.BatchNorm1d(20*40*3),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.decoder(x) 
        x = x.view(x.size(0), 3, 20, 40)
        return x

# line and legend patches flattened and concat. then sent through this n/w to classify if matching
class MLP_Relation_Net(nn.Module):
    def __init__(self):
        super(MLP_Relation_Net, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=2*20*40*3, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.network(x)
        return x

# line and legend patches are concatenated. then sent through this n/w to classify if matching
class Conv_Relation_Net(nn.Module):
    def __init__(self):
        super(Conv_Relation_Net, self).__init__()
        self.conv = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv = nn.Sequential (
            nn.Linear(in_features=64*5*10, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.conv(x)
        x1 = x1.view(x1.size(0), -1)
        x = self.classifier(x1)
        return x

# line and legend patches sent thru 2 separate branches. followed by a classifier to tell if matching
class Branched_Conv_Relation_Net(nn.Module):
    def __init__(self):
        super(Branched_Conv_Relation_Net, self).__init__()
        self.branch1 = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential (
            nn.Linear(in_features=2*5*10*64, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, legend, line):
        x1 = self.branch1(legend)
        x2 = self.branch2(line)
        x = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim = 1)#flatten and concat
        x = self.classifier(x)
        return x

# line and legend patches flattended and then sent thru 2 separate branches. followed by a classifier to tell if matching
class Branched_MLP_Relation_Net(nn.Module):
    def __init__(self):
        super(Branched_Conv_Relation_Net, self).__init__()
        self.branch1 = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential (
            nn.Linear(in_features=2*5*10*64, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, legend, line):
        x1 = self.branch1(legend)
        x2 = self.branch2(line)
        x = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim = 1)#flatten and concat
        x = self.classifier(x)
        return x

# line and legend patches sent thru 2 separate branches. followed by a classifier to tell if matching
# also, autoencoder to see if enforcing reconstruction improves anything
class Branched_Conv_Relation_Net_Autoenc(nn.Module):
    def __init__(self):
        super(Branched_Conv_Relation_Net_Autoenc, self).__init__()
        self.branch1 = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.branch2 = nn.Sequential (
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
 
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.decoder1 = nn.Sequential (
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.decoder2 = nn.Sequential (
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential (
            nn.Linear(in_features=2*5*10*64, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(in_features=32, out_features=2),
            nn.Sigmoid(),
        )

    def forward(self, legend, line):
        x1 = self.branch1(legend)
        x2 = self.branch2(line)
        legend_reconstructed = self.decoder1(x1)
        line_reconstructed = self.decoder2(x2)
        x = torch.cat((x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)), dim = 1)#flatten and concat
        x = self.classifier(x)
        return [x, legend_reconstructed, line_reconstructed]
