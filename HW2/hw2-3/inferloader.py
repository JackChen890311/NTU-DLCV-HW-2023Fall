import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from constant import CONSTANT
from dataloader import Preprocess


class FuckDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path, preprocess):
        # Initialize your dataset object
        super().__init__()
        self.data_path = data_path
        self.preprocess = preprocess
        self.file_name = sorted(os.listdir(self.data_path))
        self.df = pd.DataFrame({
            'image_name': self.file_name,
            'label': [-1] * len(self.file_name)
        })

    def __len__(self):
        # Return the length of your dataset
        return len(self.file_name)
    
    def __getitem__(self, idx):
        # Return an item pair, e1.g. dataset[idx] and its label
        img = self.preprocess(Image.open(os.path.join(self.data_path,self.df['image_name'][idx])).convert('RGB'))
        label = int(self.df['label'][idx])
        return img, label
    

class FuckDataloader():
    def __init__(self, dataset_type):
        self.C = CONSTANT()
        self.P = Preprocess()
        self.dataset_type = dataset_type
        self.loader = {}
        self.df = {}

        if self.dataset_type == 'svhn':
            self.preprocess_train = self.P.svhn_pre_train
            self.preprocess_test = self.P.svhn_pre_test
        elif self.dataset_type == 'usps':
            self.preprocess_train = self.P.svhn_pre_train
            self.preprocess_test = self.P.svhn_pre_test
        else:
            print('Error: Wrong dataset type')
            return

    def setup(self, types):
        print('Loading Data...')

        if self.dataset_type == 'svhn':
            self.data_path = self.C.svhn_path
        elif self.dataset_type == 'usps':
            self.data_path = self.C.usps_path

        mapping = {
            'train':[self.data_path, self.preprocess_train, True],
            'valid':[self.data_path, self.preprocess_test, False],
            'test' :[self.data_path, self.preprocess_test, False],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            path, preprocess, shuffle = mapping[name]
            dataset = FuckDataset(path, preprocess)
            self.loader[name] = self.loader_prepare(dataset, shuffle)
            self.df[name] = dataset.df
        
        if setupNames:
            print('Preparation Done! Use dataloader.loader[{type}] to access each loader.')
        else:
            print('Error: There is nothing to set up')

    def loader_prepare(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.C.bs,
            num_workers=self.C.nw,
            shuffle=shuffle,
            pin_memory=self.C.pm
        )
    