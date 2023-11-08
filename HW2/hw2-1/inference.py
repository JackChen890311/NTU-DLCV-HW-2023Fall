import os
import torch
import argparse
import numpy as np
from torchvision.utils import save_image, make_grid

from model import DDPM, ContextUnet
from constant import CONSTANT


def inference(C, eval_only = False):
    ddpm = DDPM(nn_model=ContextUnet(in_channels=3, n_feat=C.n_feat, n_classes=C.n_classes), betas=(1e-4, 0.02), n_T=C.n_T, device=C.device, drop_prob=0.1)
    ddpm.to(C.device)
    with open(C.model_path, 'rb') as f:
        ddpm.load_state_dict(torch.load(f))

    ddpm.eval()
    with torch.no_grad():
        # 100 = 50 x 2 (in case of OOM)
        n_sample = 50*C.n_classes
        torch.manual_seed(42069)
        x_gen, x_gen_store = ddpm.sample(n_sample, (3, 28, 28), C.device, guide_w=C.w_infer)
        torch.manual_seed(69420)
        x_gen2, x_gen_store2 = ddpm.sample(n_sample, (3, 28, 28), C.device, guide_w=C.w_infer) 

        x_gen = torch.cat([x_gen, x_gen2], dim=0)
        x_gen_store = np.concatenate([x_gen_store, x_gen_store2], axis=1)
        # print(x_gen.shape,x_gen_store.shape) # sample x 3 x 28 x 28, (n_t // 20) x sample x 3 x 28 x 28
        
        # save 50 x 2 x 10 = 1000 images
        for i in range(x_gen.shape[0]):
            save_image(x_gen[i]*-1 + 1, os.path.join(C.infer_eval_dir, f"{i%10}_%03d.png"%(i//10+1)))
        
        # plot the diffusion process and first 100 images
        if not eval_only: 
            finalCat = [torch.zeros((3, 28, 3))]
            for i in range(6):
                img_step = x_gen_store[i*5,0,:,:,:]
                finalCat.append(torch.tensor(img_step))
                finalCat.append(finalCat[0])
                save_image(torch.tensor(img_step*-1 + 1), os.path.join(C.infer_save_dir, f"process_image_{i}.png"))
            
            catImage = torch.cat(finalCat, dim=2)
            save_image(catImage*-1 + 1, os.path.join(C.infer_save_dir, 'concatenated_image.png'))
            
            x_gen = x_gen[:100]
            grid = make_grid(x_gen*-1 + 1, nrow=10)
            save_image(grid, os.path.join(C.infer_save_dir, f"image_all.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--output", type=str, help="Output image directory")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    C = CONSTANT()
    C.model_path = 'models/model_2-1.pth'
    C.infer_eval_dir = args.output
    C.infer_save_dir = ''
    inference(C, eval_only = True)