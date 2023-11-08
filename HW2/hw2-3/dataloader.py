import os
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from constant import CONSTANT


class Preprocess():
    def __init__(self):
        self.mnistm_pre_train = T.Compose([
            # T.Resize((32,32)),
            # T.RandomCrop((28,28)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.4631, 0.4666, 0.4195], [0.1979, 0.1845, 0.2083])
        ])
        self.svhn_pre_train = T.Compose([
            # T.Resize((32,32)),
            # T.RandomCrop((28,28)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.4413, 0.4458, 0.4715], [0.1169, 0.1206, 0.1042])
        ])
        self.usps_pre_train = T.Compose([
            # T.Resize((32,32)),
            # T.RandomCrop((28,28)),
            # T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.2573, 0.2573, 0.2573], [0.3373, 0.3373, 0.3373])
        ])

        self.mnistm_pre_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4631, 0.4666, 0.4195], [0.1979, 0.1845, 0.2083])
        ])
        self.svhn_pre_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.4413, 0.4458, 0.4715], [0.1169, 0.1206, 0.1042])
        ])
        self.usps_pre_test = T.Compose([
            T.ToTensor(),
            T.Normalize([0.2573, 0.2573, 0.2573], [0.3373, 0.3373, 0.3373])
        ])


class MyDataset(Dataset):
    # Implement your dataset here
    def __init__(self, data_path, preprocess, csvfiles):
        # Initialize your dataset object
        super().__init__()
        self.data_path = data_path
        self.preprocess = preprocess
        self.csvfiles = csvfiles
        self.df = pd.DataFrame()
        for csvfile in csvfiles:
            temp = pd.read_csv(csvfile)
            self.df = pd.concat([self.df, temp])
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        # Return the length of your dataset
        return len(self.df)
    
    def __getitem__(self, idx):
        # Return an item pair, e1.g. dataset[idx] and its label
        img = self.preprocess(Image.open(os.path.join(self.data_path,self.df['image_name'][idx])).convert('RGB'))
        label = int(self.df['label'][idx])
        return img, label


class MyDataloader():
    def __init__(self, dataset_type):
        self.C = CONSTANT()
        self.P = Preprocess()
        self.dataset_type = dataset_type
        self.loader = {}
        self.df = {}

        if self.dataset_type == 'mnistm':
            self.preprocess_train = self.P.mnistm_pre_train
            self.preprocess_test = self.P.mnistm_pre_test
        elif self.dataset_type == 'svhn':
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

        if self.dataset_type == 'mnistm':
            self.data_path = self.C.mnistm_path
        elif self.dataset_type == 'svhn':
            self.data_path = self.C.svhn_path
        elif self.dataset_type == 'usps':
            self.data_path = self.C.usps_path

        self.csv_train = [os.path.join(self.data_path, '../train.csv')]
        self.csv_valid = [os.path.join(self.data_path, '../val.csv')]
        self.csv_test = [os.path.join(self.data_path, '../test.csv')]

        mapping = {
            'train':[self.data_path, self.preprocess_train, self.csv_train, True],
            'valid':[self.data_path, self.preprocess_test, self.csv_valid, False],
            'test' :[self.data_path, self.preprocess_test, self.csv_test, False],
                   }
        setupNames = list(set(types) & set(mapping.keys()))
        
        for name in setupNames:
            path, preprocess, csvfiles, shuffle = mapping[name]
            dataset = MyDataset(path, preprocess, csvfiles)
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
    

class AllDataloader():
    def __init__(self):
        self.C = CONSTANT()
        self.loaders = {
            'mnistm': MyDataloader('mnistm'),
            'svhn': MyDataloader('svhn'),
            'usps': MyDataloader('usps')
        }

    def setup(self, source, target, setup_types):
        self.source_loader = self.loaders[source]
        self.target_loader = self.loaders[target]
        self.source_loader.setup(setup_types)
        self.target_loader.setup(setup_types)


if __name__ == '__main__':
    # mnistm = MyDataloader('mnistm')
    # mnistm.setup(['train', 'valid'])

    # for x,y in mnistm.loader['valid']:
    #     print(x.shape, y.shape)
    #     break

    # print(mnistm.df['valid'].head())
    # print(mnistm.df['valid'].shape)

    C = CONSTANT()
    adl = AllDataloader()
    adl.setup(C.source, C.target, ['train', 'valid'])

    # for x,y in adl.target_loader.loader['valid']:
    #     print(x.shape, y.shape)
    #     break
    
    print(adl.source_loader.df['train'].shape)
    print(adl.source_loader.df['valid'].shape)
    print(adl.target_loader.df['train'].shape)
    print(adl.target_loader.df['valid'].shape)