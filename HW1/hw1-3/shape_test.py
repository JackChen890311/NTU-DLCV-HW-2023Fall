import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from constant import CONSTANT

C = CONSTANT()

class MyModel(nn.Module):
    # Implement your model here
    def __init__(self):
        # Initialize your model object
        super(MyModel, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.preLayers = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.l1 = self.backbone.layer1 # 1/4
        self.l2 = self.backbone.layer2 # 1/8
        self.l3 = self.backbone.layer3 # 1/16
        self.l4 = self.backbone.layer4 # 1/32

        self.postLayers = nn.Sequential(
            self.backbone.avgpool,
            # self.backbone.fc
        )

        
    def forward(self, x):
        # Return the output of model given the input x
        x = self.preLayers(x)
        x = self.l1(x)
        print("Passed Layer 1: ",x.shape)
        x = self.l2(x)
        print("Passed Layer 2: ",x.shape)
        x = self.l3(x)
        print("Passed Layer 3: ",x.shape)
        x = self.l4(x)
        print("Passed Layer 4: ",x.shape)
        x = self.postLayers(x)
        return x


class MyModel(nn.Module):
    # Implement your model here
    def __init__(self):
        # Initialize your model object
        super(MyModel, self).__init__()
        ''' VGG16 '''
        self.backbone = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.backbone[0].padding = (100,100)
        # 5, 10, 17, 24, 31
        # 1/2(64), 1/4(128), 1/8(256), 1/16(512), 1/32(512)
        self.layer_0_17 = self.backbone[0:17] # 1/8
        self.layer_17_24 = self.backbone[17:24] # 1/16
        self.layer_24_31 = self.backbone[24:31] # 1/32

        '''ResNet 50'''
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.conv1.padding = (100,100)
        self.preLayers = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1, 
            self.backbone.relu,
            self.backbone.maxpool
        )
        self.l1 = self.backbone.layer1 # 1/4
        self.l2 = self.backbone.layer2 # 1/8
        self.l3 = self.backbone.layer3 # 1/16
        self.l4 = self.backbone.layer4 # 1/32

        self.postLayers = nn.Sequential(
            self.backbone.avgpool,
            self.backbone.fc
        )

        # 1x1 to adjust channels
        self.ca1 = None
        self.ca2 = nn.Conv2d(512, 256, 1)
        self.ca3 = nn.Conv2d(1024, 512, 1)
        self.ca4 = nn.Conv2d(2048, 512, 1)

        # After backbone
        # Reference: https://github.com/wkentaro/pytorch-fcn/blob/main/torchfcn/models/fcn32s.py
        self.conv = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),

            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()            
        )

        # FCN32s
        self.score_fr = nn.Conv2d(4096, C.classes, 1)
        self.upscore = nn.ConvTranspose2d(C.classes, C.classes, 64, 32, bias=False)
        
        # FCN16s
        self.score_pool4 = nn.Conv2d(512, C.classes, 1)
        self.upscore2 = nn.ConvTranspose2d(C.classes, C.classes, 4, 2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(C.classes, C.classes, 32, 16, bias=False)
        
        # FCN8s
        self.score_pool3 = nn.Conv2d(256, C.classes, 1)
        self.upscore8 = nn.ConvTranspose2d(C.classes, C.classes, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(C.classes, C.classes, 4, stride=2, bias=False)

        self.combine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(21, 512),

            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(512, 7)
        )


    def forward(self, x):
        # Return the output of model given the input x
        x_size = x.size()

        '''VGG 16'''
        f8 = self.layer_0_17(x)
        f16 = self.layer_17_24(f8)
        f32 = self.layer_24_31(f16)

        '''ResNet 50'''
        px = self.preLayers(x)
        f4 = self.l1(px)
        f8 = self.l2(f4)
        f16 = self.l3(f8)
        f32 = self.l4(f16)

        f8 = self.ca2(f8)
        f16 = self.ca3(f16)
        f32 = self.ca4(f32)

        # Post
        f32 = self.conv(f32)

        # FCN32s
        score_fr = self.score_fr(f32)
        x32 = self.upscore(score_fr)
        x32 = x32[:, :, 19:19 + x_size[2], 19:19 + x_size[3]].contiguous()
        
        # FCN16s
        upscore2 = self.upscore2(score_fr)
        score_pool4 = self.score_pool4(f16)
        score_pool4 = score_pool4[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        x16_t = upscore2 + score_pool4
        x16 = self.upscore16(x16_t)
        x16= x16[:, :, 27:27 + x_size[2], 27:27 + x_size[3]].contiguous()

        # FCN8s
        upscore_pool4 = self.upscore_pool4(x16_t)
        x8 = self.score_pool3(f8)
        x8 = x8[:, :, 9:9 + upscore_pool4.size()[2], 9:9 + upscore_pool4.size()[3]]
        x8 += upscore_pool4 
        x8 = self.upscore8(x8)
        x8 = x8[:, :, 31:31 + x_size[2], 31:31 + x_size[3]].contiguous()
        
        # print(x32.shape)
        # print(x16.shape)
        # print(x8.shape)

        # Combine them with a linear layer
        xall = torch.cat((x32, x16, x8), dim=1).permute((0, 2, 3, 1))
        # print(xall.shape)
        xall = self.combine(xall).permute((0, 3, 1, 2))

        return xall
    

if __name__ == '__main__':
    model = MyModel().to(C.device)
    print(model)

    dummy = torch.randn((4, 3, 512, 512))
    print("Original Shape: ", dummy.shape)
    dummy = dummy.to(C.device)
    yhat = model(dummy)
    print("Final Shape: ", yhat.shape)