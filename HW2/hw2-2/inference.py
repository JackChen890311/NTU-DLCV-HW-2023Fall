import os
import argparse
import torch
from torchvision.utils import save_image

from constant import CONSTANT
from UNet import UNet
from functions import sample_sequence

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, help="Input noise directory")
    parser.add_argument("--output", type=str, help="Output image directory")
    parser.add_argument("--model_path", type=str, help="model path")
    args = parser.parse_args()
    print(args)

    C = CONSTANT()

    model = UNet()
    with open(args.model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    model = model.to(C.device)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    noises = []
    noisename = sorted(os.listdir(args.input))
    for name in noisename:
        with open(os.path.join(args.input, name), 'rb') as f:
            noises.append(torch.load(f))
    noises = torch.cat(noises).float()
    noises = noises.to(C.device)

    # Given 10 noises
    print("Starting sampling using given noises...")
    gtCat = []
    mse_total = 0
    sample_img = sample_sequence(model, noises, C.eta) # [batch x 3 x h x w] * timesteps (list)
    for j in range(sample_img[0].size(0)):
        save_image(sample_img[-1][j], os.path.join(args.output, noisename[j].split('.')[0]+'.png'))
