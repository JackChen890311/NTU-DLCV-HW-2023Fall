#!/bin/bash

gdown -O model1-1.pt 'https://drive.google.com/uc?id=1j74JTY-FeNN-OCWGeOaYGtqildn5R-dI'
gdown -O model1-2.pt 'https://drive.google.com/uc?id=16KW0wBC955eh3an5cgYzDe7R24G6bD1u'
gdown -O model1-3.pt 'https://drive.google.com/uc?id=17THy45jhIo9-Xm-kTkfx9RHc3tDr1L34'
python3 -c "import torchvision.models as models; \
    from torchvision.models.segmentation import deeplabv3_resnet101; \
    model=deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT);"
