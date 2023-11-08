''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import  transforms
from torchvision.utils import save_image, make_grid

from model import DDPM, ContextUnet
from constant import CONSTANT
from dataloader import MyDataset


def train_mnist():
    
    C = CONSTANT()
    n_epoch = C.epochs
    batch_size = C.bs
    n_T = C.n_T
    device = C.device
    n_classes = C.n_classes
    n_feat = C.n_feat
    lrate = C.lr
    nw = C.nw

    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    save_model = C.save_model
    save_dir = 'output/' + start_time + '/train/'

    if not os.path.exists('output'):
        os.mkdir('output')
    os.mkdir('output/%s'%start_time)
    os.mkdir('output/%s/train'%start_time)
    with open('output/%s/log.txt'%start_time, 'w') as f:
        for key, value in vars(C).items():
            f.write(f"{key}: {value}\n")
    
    ws_test = [0.0, 0.5, 2.0] # strength of generative guidance

    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset = MyDataset(C.data_path, tf, [C.csv_path_train, C.csv_path_test])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)

    for ep in tqdm(range(n_epoch)):
        print(f'epoch {ep}')
        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch)

        pbar = tqdm(dataloader)
        loss_ema = None
        for x, c in pbar:
            optim.zero_grad()
            x = x.to(device)
            c = c.to(device)
            loss = ddpm(x, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
        
        # for eval, save an image of currently generated samples (top rows)
        # followed by real images (bottom rows)
        if ep % C.verbose==0 or ep == int(n_epoch-1):
            ddpm.eval()
            with torch.no_grad():
                n_sample = 10*n_classes
                for w_i, w in enumerate(ws_test):
                    x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28, 28), device, guide_w=w)              

                    grid = make_grid(x_gen*-1 + 1, nrow=10)
                    save_image(grid, save_dir + f"image_ep{ep}_w{w}.png")
                    print('saved image at ' + save_dir + f"image_ep{ep}_w{w}.png")

        # optionally save model
        if save_model and ep == int(n_epoch-1):
            torch.save(ddpm.state_dict(), save_dir + f"model_{ep}.pth")
            print('saved model at ' + save_dir + f"model_{ep}.pth")

if __name__ == "__main__":
    train_mnist()
