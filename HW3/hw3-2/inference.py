import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT
from model import MyModel
from tokenizer import BPETokenizer

C = CONSTANT()

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

# amp_enable = False
# amp_bf16 = True
# amp_dtype = torch.bfloat16 if amp_bf16 else torch.float16
# amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Preprocess():
    def __init__(self):
        self.test_pre = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class TestDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path, preprocess):
        # Initialize your dataset object
        super().__init__()
        self.data_path = data_path
        self.preprocess = preprocess
        self.filename = os.listdir(self.data_path)

    def __len__(self):
        # Return the length of your dataset
        return len(self.filename)
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        image = Image.open(os.path.join(self.data_path, self.filename[idx])).convert('RGB')
        image = self.preprocess(image)
        return image, -1


class TestDataloader():
    def __init__(self, data_path):
        self.data_path = data_path
        self.C = CONSTANT()
        self.P = Preprocess()
        self.loader = {}
        self.filename = {}

    def setup(self, types):
        print('(Testloader) Loading Data...')

        mapping = {
            'test' :[self.data_path, self.P.test_pre, False, self.C.bs_infer],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            data, preprocess, shuffle, batch_size = mapping[name]
            dataset = TestDataset(data, preprocess)
            self.loader[name] = self.loader_prepare(dataset, shuffle, batch_size)
            self.filename[name] = dataset.filename
            del dataset
        
        if setupNames:
            print('(Testloader) Preparation Done! Use dataloader.loader[{type}] to access each loader.')
        else:
            print('(Testloader) Error: There is nothing to set up')

    def loader_prepare(self, dataset, shuffle, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.C.nw,
            shuffle=shuffle,
            pin_memory=self.C.pm
        )


def inference(args):
    # Model
    model = MyModel(encoder_name = C.encoder_model_name, decoder_path = args.decoder_path)

    if args.model_type == 'valid':
        raise NotImplementedError
        # state = model.load_state_dict(torch.load(os.path.join(args.model_path,'trainable_weights.pt')), strict=False)
    elif args.model_type == 'cider':
        state = model.load_state_dict(torch.load(os.path.join(args.model_path,'trainable_weights_cider.pt')), strict=False)
    elif args.model_type == 'clip':
        raise NotImplementedError
        # state = model.load_state_dict(torch.load(os.path.join(args.model_path,'trainable_weights_clip.pt')), strict=False)
    else:
        raise NotImplementedError
    assert len(state.unexpected_keys) == 0
    model = model.to(C.device)
    model.eval()

    # Data
    dataloaders = TestDataloader(args.folder)
    dataloaders.setup(['test'])

    # Add your own metrics here
    loader = dataloaders.loader['test']
    filename = dataloaders.filename['test']

    # Run
    result = run_inference(model, loader, filename, args.mode)

    with open(args.output_json, 'w') as f:
        json.dump(result, f, indent = 4)
    
    
def run_inference(model, loader, filename, mode):
    filecnt = 0
    result = {}
    try:
        tokenizer = BPETokenizer('encoder.json', 'vocab.bpe')
    except:
        tokenizer = BPETokenizer('hw3-2/encoder.json', 'hw3-2/vocab.bpe')

    with torch.no_grad():
        for img,_ in tqdm(loader):
            batch_cnt = img.size(0)
            img = img.to(C.device)

            # with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enable):
            if mode == 'greedy':
                raise NotImplementedError
                # yhat = model.autoreg_infer(img, step=C.infer_step_greedy)
            elif mode == 'beam':
                yhat = model.autoreg_infer_beam(img, beam=C.beam_size, step=C.infer_step_beam)
            else:
                raise NotImplementedError

            if C.PEFT_mode == 'prefix':
                yhat = yhat[:, C.prefix_cnt:]

            for i in range(batch_cnt):
                name = filename[filecnt].split('.')[0]

                yhat_sub = yhat[i].detach().cpu().tolist()[1:]
                try:
                    y_hat_endidx = yhat_sub.index(50256)
                except:
                    y_hat_endidx = len(yhat_sub)
                yhat_sub = yhat_sub[:y_hat_endidx]

                predict = tokenizer.decode(yhat_sub)
                
                filecnt += 1
                result[name] = predict
    
    print(len(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--model_type', type=str, default='')
    parser.add_argument('--folder', type=str, default='')
    parser.add_argument('--output_json', type=str, default='')
    parser.add_argument('--decoder_path', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args()

    inference(args)