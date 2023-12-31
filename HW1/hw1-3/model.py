import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead

from constant import CONSTANT
from dataloader import MyDataloader

C = CONSTANT()

# class MyModel(nn.Module):
#     # Implement your model here
#     def __init__(self):
#         # Initialize your model object
#         super(MyModel, self).__init__()
#         # self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
#         self.backbone = models.vgg16(weights=None).features
#         # Source: https://github.com/sairin1202/fcn32-pytorch/blob/master/pytorch-fcn32.py 
#         # Reference: https://github.com/wkentaro/pytorch-fcn/blob/main/torchfcn/models/fcn32s.py
#         self.backbone[0].padding=(100,100)
#         self.fcn32s = nn.Sequential(
#             nn.Conv2d(512, 4096, 7),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(),

#             nn.Conv2d(4096, 4096, 1),
#             nn.ReLU(inplace=True),
#             nn.Dropout2d(),

#             nn.Conv2d(4096, C.classes, 1),
#             nn.ConvTranspose2d(C.classes, C.classes, 64, 32, bias=False)
#             )
        

#     def forward(self, x):
#         # Return the output of model given the input x
#         x_size = x.size()
#         print(x.shape)
#         x = self.backbone(x)
#         print(x.shape)
#         x = self.fcn32s(x)
#         print(x.shape)
#         x = x[:, :, 19:19 + x_size[2], 19:19 + x_size[3]].contiguous()
#         return x


class MyModel(nn.Module):
    # Implement your model here
    def __init__(self):
        # Initialize your model object
        super(MyModel, self).__init__()
        # Reference: https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
        self.backbone = deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
        self.backbone.classifier = DeepLabHead(2048, C.classes)
        self.backbone.aux_classifier = FCNHead(1024, C.classes)

    def forward(self, x):
        return self.backbone(x)['out']


if __name__ == '__main__':
    model = MyModel().to(C.device)
    print(str(model))
    
    dataloaders = MyDataloader()
    dataloaders.setup(['valid'])

    for x,y in dataloaders.loader['valid']:
        x = x.to(C.device)
        xx = {'aux':x, 'out':x}
        yhat = model(x)
        print(x.shape, y.shape)
        print(yhat.shape)
        break