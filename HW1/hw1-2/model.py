import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from constant import CONSTANT
from dataloader import MyDataloader

C = CONSTANT()

class MyModel(nn.Module):
    # Implement your model here
    def __init__(self):
        # Initialize your model object
        super(MyModel, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.dense = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Linear(1000, 1000),
            nn.Dropout(),
            nn.LeakyReLU(),
            
            nn.Linear(1000, 1000),
            nn.Dropout(),
            nn.LeakyReLU(),

            nn.Linear(1000, C.classes)
        )
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

        self.dense.apply(init_weights)


    def forward(self, x):
        # Return the output of model given the input x
        x = self.backbone(x)
        x = F.relu(x)
        x = self.dense(x)
        return x


if __name__ == '__main__':
    C = CONSTANT()
    model = MyModel().to(C.device)
    print(model)
    
    dataloaders = MyDataloader()
    dataloaders.setup(['valid'])

    for x,y in dataloaders.loader['valid']:
        x = x.to(C.device)
        y = y.to(C.device)
        yhat = model(x)
        print(x.shape,y.shape)
        print(yhat.shape)
        break