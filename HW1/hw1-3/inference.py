import os
import torch
import argparse
from torchvision.utils import save_image


from model import MyModel
from dataloader import MyDataloader
from constant import CONSTANT
from main import idxtoRGB
# from mean_iou_evaluate import read_masks, mean_iou_score

C = CONSTANT()

def inference(args):
    # Model
    model = MyModel()
    model.load_state_dict(torch.load(C.model_path_final))
    model = model.to(C.device)
    model.eval()

    # Data
    dataloaders = MyDataloader()
    dataloaders.C.data_path_test = args.input
    dataloaders.setup(['test'])
    loader = dataloaders.loader['test']
    filename = dataloaders.filenames['test']
    output_path = args.output
    # gt_path = args.input

    # test
    cnt = 0
    for x,y in loader:
        x = x.to(C.device)
        yhat = model(x)
        # for miou
        yidx = torch.argmax(yhat, dim = 1)
        for i in range(x.shape[0]):
            rgbimg = idxtoRGB(yidx[i].squeeze())
            save_image(rgbimg, os.path.join(output_path, filename[cnt]+'_mask.png'))
            cnt += 1

    # pred = read_masks(output_path)
    # labels = read_masks(gt_path)
    # miou = mean_iou_score(pred, labels)
    # print('MIOU: ', miou)
    
    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argparse usage")
    parser.add_argument("--input", help="Input image directory")
    parser.add_argument("--output", help="Output image directory")
    args = parser.parse_args()
    print(args)
    
    inference(args)