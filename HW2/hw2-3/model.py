import torch.nn as nn
from utils import ReverseLayerF
from constant import CONSTANT
from dataloader import AllDataloader


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),

        )

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(-1, 512)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),

            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),

            nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(in_features=512, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        return x
    
    
if __name__ == '__main__':
    C = CONSTANT()
    adl = AllDataloader()
    adl.setup(C.source, C.target, ['train', 'valid'])
    
    e = Extractor()
    c = Classifier()
    d = Discriminator()

    for x,y in adl.target_loader.loader['train']:
        print(x.shape, y.shape)

        e_out = e(x)
        print(e_out.shape)

        c_out = c(e_out)
        print(c_out.shape) 

        d_out = d(e_out, 0.5)
        print(d_out.shape)
        break
    