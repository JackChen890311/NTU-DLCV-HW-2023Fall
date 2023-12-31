import torch
import argparse

import model
from inferloader import FuckDataloader
from constant import CONSTANT


def inference(encoder, classifier, target, df, csv_path):
    all_pred = []
    for x, y in target:
        x,y = x.to(C.device), y.to(C.device)
        emb = encoder(x)
        pred = classifier(emb)
        all_pred.append(pred.detach().cpu())

    all_pred = torch.argmax(torch.cat(all_pred, dim=0), dim=1)

    df['label'] = all_pred.numpy()
    df.to_csv(csv_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, help="Input image directory")
    parser.add_argument("--output", type=str, help="Output csv file path")
    args = parser.parse_args()
    print(args)
    
    if 'svhn' in args.input:
        target = FuckDataloader('svhn')
        C = CONSTANT(target='svhn')
        target.C.svhn_path = args.input
    elif 'usps' in args.input:
        target = FuckDataloader('usps')
        C = CONSTANT(target='usps')
        target.C.usps_path = args.input

    target.setup(['test'])

    with torch.no_grad():
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        with open(C.final_encoder_path,'rb') as f:
            encoder.load_state_dict(torch.load(f))
        with open(C.final_classfier_path,'rb') as f:
            classifier.load_state_dict(torch.load(f))
        inference(encoder, classifier, target.loader['test'], target.df['test'], args.output)