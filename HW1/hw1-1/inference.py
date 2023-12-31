import torch
import argparse
import pandas as pd 

from model import MyModel
from dataloader import MyDataloader
from constant import CONSTANT

C = CONSTANT()

def inference(args):
    model = MyModel()
    model.load_state_dict(torch.load(C.model_path_final))
    model = model.to(C.device)

    model.eval()
    dataloaders = MyDataloader()
    dataloaders.C.data_path_test = args.input
    dataloaders.setup(['test'])
    loader = dataloaders.loader['test']
    filenames = dataloaders.filenames['test']

    # accu, cnt = 0, 0
    prediction = torch.zeros((0,1))
    for x,y in loader:
        x = x.to(C.device)
        yhat = model(x)
        y_idx = torch.argmax(yhat, 1)
        prediction = torch.cat((prediction,y_idx.unsqueeze(1).to('cpu')))
        # accu += torch.sum(y_idx == y).item()
        # cnt += len(y)
    # print('ACCU:',accu/cnt)
    
    csv_data = {
        'filename': filenames,
        'label': prediction[:,0].int()
    }
    df = pd.DataFrame(csv_data)
    df.to_csv(args.output, index=False)
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple script to demonstrate argparse usage")
    parser.add_argument("--input", help="Input image directory")
    parser.add_argument("--output", help="Output csv file path")
    args = parser.parse_args()
    print(args)
    
    inference(args)