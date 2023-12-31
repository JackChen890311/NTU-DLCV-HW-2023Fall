import os
import math
import torch
import argparse
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT
from model import MyModel
from tokenizer import BPETokenizer

C = CONSTANT()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

amp_enable = False
amp_bf16 = True
amp_dtype = torch.bfloat16 if amp_bf16 else torch.float16
amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Preprocess():
    def __init__(self):
        self.test_tensor = T.Compose([
            T.ToTensor(),
        ])
        self.test_pre = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class TestDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path):
        # Initialize your dataset object
        super().__init__()
        self.P = Preprocess()
        self.data_path = data_path
        self.filename = os.listdir(self.data_path)
        self.origin_img = {}
        for name in self.filename:
            image = Image.open(os.path.join(self.data_path, name)).convert('RGB')
            self.origin_img[name] = self.P.test_tensor(image)


    def __len__(self):
        # Return the length of your dataset
        return len(self.filename)
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        image = Image.open(os.path.join(self.data_path, self.filename[idx])).convert('RGB')
        image = self.P.test_pre(image)
        return image, -1


class TestDataloader():
    def __init__(self, data_path):
        self.data_path = data_path
        self.C = CONSTANT()
        self.loader = {}
        self.filename = {}
        self.origin_img = {}

    def setup(self, types):
        print('(Testloader) Loading Data...')

        mapping = {
            'test' :[self.data_path, False, self.C.bs_infer],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            data, shuffle, batch_size = mapping[name]
            dataset = TestDataset(data)
            self.loader[name] = self.loader_prepare(dataset, shuffle, batch_size)
            self.filename[name] = dataset.filename
            self.origin_img[name] = dataset.origin_img
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
    model = MyModel(encoder_name = C.encoder_model_name, decoder_path = args.decoder_path, save_map=True)

    if args.model_type == 'valid':
        state = model.load_state_dict(torch.load(os.path.join(args.model_path,'trainable_weights.pt')), strict=False)
    elif args.model_type == 'cider':
        state = model.load_state_dict(torch.load(os.path.join(args.model_path,'trainable_weights_cider.pt')), strict=False)
    elif args.model_type == 'clip':
        state = model.load_state_dict(torch.load(os.path.join(args.model_path,'trainable_weights_clip.pt')), strict=False)
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
    origin_img = dataloaders.origin_img['test']

    # mkdir
    os.makedirs(args.output_path, exist_ok=True)
    if os.path.exists('attn_map.pkl'):
        os.remove('attn_map.pkl')

    # Run
    result = run_inference(model, loader, filename, origin_img, args.mode)
    
    
def run_inference(model, loader, filename, origin_img, mode):
    filecnt = 0
    result = {}
    tokenizer = BPETokenizer('encoder.json', 'vocab.bpe')

    with torch.no_grad():
        for img,_ in tqdm(loader):
            batch_cnt = img.size(0)
            img = img.to(C.device)

            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enable):
                if mode == 'greedy':
                    yhat = model.autoreg_infer(img, step=C.infer_step_greedy)
                elif mode == 'beam':
                    raise NotImplementedError
                else:
                    raise NotImplementedError

            if C.PEFT_mode == 'prefix':
                yhat = yhat[:, C.prefix_cnt:]
            
            with open('attn_map.pkl', 'rb') as f:
                att = pk.load(f)

            for i in range(batch_cnt):
                rgb_img = origin_img[filename[filecnt]]
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
                ratio = rgb_img.shape[1]/rgb_img.shape[2]
                fig = plt.figure(figsize=(16, 16*ratio))

                fig.add_subplot(math.ceil((len(yhat_sub)+1)/5), 5, 1)
                plt.imshow(rgb_img.numpy().transpose(1, 2, 0))
                plt.title('Original')
                plt.axis('off')
                
                for j in range(len(yhat_sub)):
                    att_img = att[i,0,j,1:].detach().cpu().numpy()
                    att_img = (att_img - np.min(att_img)) / (np.max(att_img) - np.min(att_img))
                    att_img = np.clip(att_img, 0, 1).reshape(16, 16)
                    att_img = resize(att_img, (rgb_img.shape[1], rgb_img.shape[2]), order=3)
                    att_img = np.expand_dims(att_img, axis=0)
                    fig.add_subplot(math.ceil((len(yhat_sub)+1)/5), 5, j+2)
                    plt.imshow(rgb_img.numpy().transpose(1, 2, 0))
                    plt.imshow((att_img).transpose(1, 2, 0), alpha = 0.5, cmap='jet')
                    plt.title(tokenizer.decode([yhat_sub[j]]))
                    plt.axis('off')
                    # att_img.save(os.path.join(args.output_path, f'{i}_{j}_{tokenizer.decode([yhat_sub[j]])}.png'))
                plt.savefig(os.path.join(args.output_path, f'{name}_attn_map.png'))
    
    print(result)
    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='greedy')
    parser.add_argument('--model_type', type=str, default='cider')
    parser.add_argument('--folder', type=str, default='../hw3_data/p3_data/images')
    parser.add_argument('--decoder_path', type=str, default='../hw3_data/p2_data/decoder_model.bin')
    parser.add_argument('--model_path', type=str, default='output/lora32/')
    parser.add_argument('--output_path', type=str, default='output/attn_map/')
    args = parser.parse_args()

    inference(args)