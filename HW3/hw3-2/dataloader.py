import os
import json
import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT
from tokenizer import BPETokenizer

C = CONSTANT()

class Preprocess():
    def __init__(self):
        self.train_pre = T.Compose([
            T.Resize([224, 224]),
            T.RandomHorizontalFlip(),
            T.RandomRotation(10),
            T.RandomGrayscale(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.test_pre = T.Compose([
            T.Resize([224, 224]),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class MyDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path, json_path, preprocess):
        # Initialize your dataset object
        super().__init__()
        self.data_path = data_path
        self.json_path = json_path
        self.preprocess = preprocess

        with open(self.json_path) as f:
            json_data = json.load(f)
        
        id_path = {str(item['id']): item['file_name'] for item in json_data['images']}
        data = [
            [item['caption'] for item in json_data['annotations']],
            [id_path[str(item['image_id'])] for item in json_data['annotations']],
        ]
        self.df = pd.DataFrame({'caption': data[0], 'image_path': data[1]})
        self.tokenizer = BPETokenizer('encoder.json', 'vocab.bpe')

        # Debug mode
        # if 'train' in data_path:
        #     self.df = self.df[:300]
        # elif 'val' in data_path:
        #     self.df = self.df[:300]

    def __len__(self):
        # Return the length of your dataset
        return len(self.df)
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        image = Image.open(os.path.join(self.data_path, self.df['image_path'][idx])).convert('RGB')
        image = self.preprocess(image)
        caption = self.df['caption'][idx]
        caption_id = self.tokenizer.encode(caption)
        if C.PEFT_mode == 'prefix':
            caption_input = list(range(C.prefix_cnt)) + [50256] + caption_id + [50256] * (C.max_seqlen - len(caption_id) - 1 - C.prefix_cnt)
            caption_decode = [-100] * C.prefix_cnt + caption_id + [50256] + [-100] * (C.max_seqlen - len(caption_id) - 1 - C.prefix_cnt)
        else:
            caption_input = [50256] + caption_id + [50256] * (C.max_seqlen - len(caption_id) - 1)
            caption_decode = caption_id + [50256] + [-100] * (C.max_seqlen - len(caption_id) - 1)
        return image, torch.tensor(caption_input), torch.tensor(caption_decode)


class MyDataloader():
    def __init__(self):
        self.C = CONSTANT()
        self.P = Preprocess()
        self.loader = {}
        self.df = {}

    def setup(self, types):
        print('Loading Data...')

        mapping = {
            'train':[self.C.data_path_train, self.C.json_path_train, self.P.train_pre, True, self.C.bs_train],
            'valid':[self.C.data_path_valid, self.C.json_path_valid, self.P.test_pre, False, self.C.bs_valid],
            'test' :[self.C.data_path_test, self.C.json_path_test, self.P.test_pre, False, self.C.bs_valid],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            data, json, preprocess, shuffle, batch_size = mapping[name]
            dataset = MyDataset(data, json, preprocess)
            self.loader[name] = self.loader_prepare(dataset, shuffle, batch_size)
            self.df[name] = dataset.df
            del dataset
        
        if setupNames:
            print('Preparation Done! Use dataloader.loader[{type}] to access each loader.')
        else:
            print('Error: There is nothing to set up')

    def loader_prepare(self, dataset, shuffle, batch_size):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.C.nw,
            shuffle=shuffle,
            pin_memory=self.C.pm
        )


if __name__ == '__main__':
    C = CONSTANT()

    dataloaders = MyDataloader()
    dataloaders.setup(['test'])
    maxlen = 0

    for img,txt,txt_decode in dataloaders.loader['test']:
        print(img.shape, txt.shape, txt_decode.shape)
        print('Input:',txt[0,0:30])
        print('Output:',txt_decode[0,0:30])
        break
        # for c in range(img.size(0)):
        #     idx = txt_decode[c].tolist().index(50256)
        #     if idx > maxlen:
        #         maxlen = idx
        # print('Max caption length:', maxlen) # 57, 55