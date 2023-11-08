import os
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from constant import CONSTANT


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


if __name__ == '__main__':
    C = CONSTANT()

    tf = T.Compose([T.ToTensor()]) # mnist is already normalised 0 to 1
    dataset = MyDataset(C.data_path, tf, [C.csv_path_train, C.csv_path_test])
    dataloader = DataLoader(dataset, batch_size=C.bs, shuffle=True, num_workers=C.nw)

    for x,y in dataloader:
        print(x.shape, y.shape)
        break

    print(dataset.df.head())
    print(dataset.df.shape)