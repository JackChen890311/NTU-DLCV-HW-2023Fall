import os
import random
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT

class Preprocess():
    def __init__(self):
        self.train_pre = T.Compose([
            T.Resize([160, 160]),
            T.RandomCrop(128),
            T.RandomHorizontalFlip(),
            T.RandomGrayscale(p=0.5),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.test_pre = T.Compose([
            T.Resize([160, 160]),
            T.CenterCrop(128),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class MiniDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path, preprocess, mode):
        # Initialize your dataset object
        self.data_path = data_path
        self.filenames = os.listdir(data_path)
        self.preprocess = preprocess
        random.shuffle(self.filenames)
        if mode == 'train':
            self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if i % 10 != 0]
        else:
            self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if i % 10 == 0]

    def __len__(self):
        # Return the length of your dataset
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        img = self.preprocess(Image.open(os.path.join(self.data_path,self.filenames[idx])).convert('RGB'))
        return img, -1


class OfficeDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path, preprocess):
        # Initialize your dataset object
        self.data_path = data_path
        self.filenames = sorted(os.listdir(data_path))
        self.preprocess = preprocess

    def __len__(self):
        # Return the length of your dataset
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Return an item pair, e.g. dataset[idx] and its label
        img = self.preprocess(Image.open(os.path.join(self.data_path,self.filenames[idx])).convert('RGB'))
        try:
            label = int(self.filenames[idx].split('_')[0])
        except:
            label = -1
        return img, label


class MyDataloader():
    def __init__(self):
        super().__init__()
        self.C = CONSTANT()
        self.P = Preprocess()
        self.loader = {}
        self.filenames = {}

    def setup(self, types):
        print('Loading Data...')

        mapping = {
            'train_ssl':[self.C.data_path_train_ssl, self.P.train_pre, True],
            'valid_ssl':[self.C.data_path_train_ssl, self.P.train_pre, True],
            'train':[self.C.data_path_train, self.P.train_pre, True],
            'valid':[self.C.data_path_valid, self.P.test_pre, False],
            'test' :[self.C.data_path_test, self.P.test_pre, False],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            path, preprocess, shuffle = mapping[name]
            if name == 'train_ssl':
                dataset = MiniDataset(path, preprocess, 'train')
                self.loader[name] = self.loader_prepare(dataset, shuffle)
                self.filenames[name] = dataset.filenames
            elif name == 'valid_ssl':
                dataset = MiniDataset(path, preprocess, 'valid')
                self.loader[name] = self.loader_prepare(dataset, shuffle)
                self.filenames[name] = dataset.filenames
            else:
                dataset = OfficeDataset(path, preprocess)
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
    print(vars(Preprocess()))
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'valid'])
    print(len(dataloaders.loader['train']),len(dataloaders.loader['valid']))

    for x,y in dataloaders.loader['valid']:
        print(x.shape, y.shape)
        print(y[:5])
        break