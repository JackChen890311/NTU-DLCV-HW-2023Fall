#!bin/bash
echo 'Downloading models...'
python3 -c "import clip; clip.load('ViT-B/32')"
python3 -c "import clip; clip.load('ViT-L/14')"
python3 -c "import timm; timm.create_model('vit_large_patch14_clip_224', pretrained=True)"
mkdir model3-2/
gdown -O model3-2/trainable_weights_cider.pt 1oMph68Dftcu6-EMCINMSKQT7B4DDrxFK