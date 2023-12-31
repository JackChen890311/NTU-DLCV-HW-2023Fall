import os
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT

class MyDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path):
        # Initialize your dataset object
        self.data_path = data_path
        self.satFiles= sorted([file for file in os.listdir(data_path) if 'sat' in file])
        self.maskFiles= sorted([file for file in os.listdir(data_path) if 'mask' in file])
        self.filenames = sorted([file.split('_')[0] for file in os.listdir(data_path) if 'mask' in file])
        self.rgb_pre = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.idx_pre = T.ToTensor()

    def __len__(self):
        # Return the length of your dataset
        assert len(self.satFiles) == len(self.maskFiles) and len(self.maskFiles) == len(self.filenames)
        return len(self.maskFiles)
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        x = self.rgb_pre(Image.open(os.path.join(self.data_path,self.satFiles[idx])))
        y = self.idx_pre(Image.open(os.path.join(self.data_path,self.maskFiles[idx])))
        y = self.rgbToLabel(y)
        return x, y
    
    def rgbToLabel(self, y):
        weights = torch.tensor([4, 2, 1])
        weighted_tensor = y * weights.view(3, 1, 1)
        weighted_sum_tensor = weighted_tensor.sum(dim=0)
        masks = torch.zeros(y.shape[1], y.shape[2]).to(torch.long)
        masks[weighted_sum_tensor == 3] = 0  # (Cyan: 011) Urban land 
        masks[weighted_sum_tensor == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[weighted_sum_tensor == 5] = 2  # (Purple: 101) Rangeland 
        masks[weighted_sum_tensor == 2] = 3  # (Green: 010) Forest land 
        masks[weighted_sum_tensor == 1] = 4  # (Blue: 001) Water 
        masks[weighted_sum_tensor == 7] = 5  # (White: 111) Barren land 
        masks[weighted_sum_tensor == 0] = 6  # (Black: 000) Unknown 
        return masks
    

class TestDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path):
        # Initialize your dataset object
        print('Using TestDataset...')
        self.data_path = data_path
        self.satFiles= sorted([file for file in os.listdir(data_path)])
        self.filenames = sorted([file.split('_')[0] for file in os.listdir(data_path)])
        self.rgb_pre = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # Return the length of your dataset
        return len(self.satFiles)
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        x = self.rgb_pre(Image.open(os.path.join(self.data_path,self.satFiles[idx])))
        y = torch.zeros(x.shape[1], x.shape[2])
        return x, y
    

class MyDataloader():
    def __init__(self):
        super().__init__()
        self.loader = {}
        self.C = CONSTANT()
        self.filenames = {}

    def setup(self, types):
        print('Loading Data...')

        mapping = {
            'train':[self.C.data_path_train, True],
            'valid':[self.C.data_path_valid, False],
            'test' :[self.C.data_path_test, False],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            path, shuffle = mapping[name]
            if name == 'test':
                dataset = TestDataset(path)
            else:
                dataset = MyDataset(path)
            self.loader[name] = self.loader_prepare(dataset, shuffle)
            self.filenames[name] = dataset.filenames
        
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


if __name__ == '__main__':
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])

    for x,y in dataloaders.loader['test']:
        print(x.shape, y.shape)
        print(torch.min(y), torch.max(y))
        break