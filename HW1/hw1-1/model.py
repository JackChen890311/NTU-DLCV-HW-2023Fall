import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MyModel(nn.Module):
    # Implement your model here
    def __init__(self):
        # Initialize your model object
        super(MyModel, self).__init__()
        # self.backbone = models.alexnet(weights=None)
        # self.backbone = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
        self.backbone = models.efficientnet_v2_m(weights=None)
        self.linear1 = nn.Linear(1000, 50)


    def forward(self, x):
        # Return the output of model given the input x
        x = F.relu(self.backbone(x))
        x = self.linear1(x)        
        return x 


if __name__ == '__main__':
    from constant import CONSTANT
    C = CONSTANT()
    model = MyModel().to(C.device)
    print(str(model))

    from dataloader import MyDataloader
    dataloaders = MyDataloader()
    dataloaders.setup(['valid'])

    for x,y in dataloaders.loader['valid']:
        x = x.to(C.device)
        yhat = model(x.float())
        print(x.shape,y.shape)
        print(yhat.shape)
        print(yhat[0])
        break