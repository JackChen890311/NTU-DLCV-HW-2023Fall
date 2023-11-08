import os
import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image, make_grid
import torchvision.transforms.functional as F

from constant import CONSTANT
from UNet import UNet
from functions import sample_sequence, sample_interpolation

if __name__ == '__main__':
    C = CONSTANT()

    model = UNet()
    with open(C.model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    model = model.to(C.device)

    if not os.path.exists(C.out_path):
        os.makedirs(C.out_path)

    noises = []
    for noisename in sorted(os.listdir(C.noise_path)):
        with open(C.noise_path + noisename, 'rb') as f:
            noises.append(torch.load(f))
    noises = torch.cat(noises).float()
    noises = noises.to(C.device)
    # print(noises.shape) # 10 x 3 x h x w

    # Given 10 noises
    print("Starting sampling using given noises...")
    gtCat = []
    mse_total = 0
    sample_img = sample_sequence(model, noises, C.eta) # [batch x 3 x h x w] * timesteps (list)
    for j in range(sample_img[0].size(0)):
        save_image(sample_img[-1][j], os.path.join(C.out_path, f"%02d.png"%j))
        # Calculate MSE between GT and sampled images
        gt = Image.open(os.path.join(C.gt_path, f"%02d.png"%j))
        sample = sample_img[-1][j].cpu()
        truth = F.to_tensor(gt) 
        gtCat.append(truth.unsqueeze(0))
        # print(sample, truth)
        mse = torch.nn.functional.mse_loss(sample * 255, truth * 255)
        mse_total += mse.item()
    gtCat = torch.cat(gtCat, dim=0)
    grid = make_grid(gtCat, nrow=gtCat.size(0))
    save_image(grid, C.out_path + '10_gt.png')

    grid = make_grid(sample_img[-1], nrow=sample_img[0].size(0))
    save_image(grid, os.path.join(C.out_path, f"10_noises.png"))
    print("Average MSE:", mse_total / sample_img[0].size(0))


    # Interpolate between 2 noises
    print("Starting interpolation...")
    interpolated_img = sample_interpolation(model, noises[0].unsqueeze(0), noises[1].unsqueeze(0), C.eta, use_slerp=True) # [11 x 3 x h x w]
    grid = make_grid(interpolated_img, nrow=interpolated_img.size(0))
    save_image(grid, os.path.join(C.out_path, f"inter_slerp.png"))

    interpolated_img = sample_interpolation(model, noises[0].unsqueeze(0), noises[1].unsqueeze(0), C.eta, use_slerp=False) # [11 x 3 x h x w]
    grid = make_grid(interpolated_img, nrow=interpolated_img.size(0))
    save_image(grid, os.path.join(C.out_path, f"inter_linear.png"))


    # Eta experiment
    print('Trying out different eta...')
    eta_list = np.arange(0,1.1,0.25)
    print(eta_list)
    for eta in eta_list:
        sample_img = sample_sequence(model, noises, eta) # [batch x 3 x h x w] * timesteps (list)

        grid = make_grid(sample_img[-1][:4], nrow=sample_img[0][:4].size(0))
        save_image(grid, os.path.join(C.out_path, f"eta_{eta}.png"))
    
    print('Done!')