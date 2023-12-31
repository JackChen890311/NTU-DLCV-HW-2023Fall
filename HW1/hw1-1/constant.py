from torch import device

class CONSTANT():
    def __init__(self):
        self.epochs = 200
        self.lr = 1e-1
        self.wd = 1e-3
        self.bs = 2
        self.nw = 4
        self.pm = True
        self.milestones = [10,50,500,5000]
        self.gamma = 0.5
        self.patience = 30
        self.verbose = 10
        self.device = device('cuda:0')

        # self.data_path = '../hw1_data/p1_data/train_50'
        # self.data_path_valid = '../hw1_data/p1_data/val_50'
        # self.data_path_test = ''
        self.data_path = ''
        self.data_path_valid = ''
        self.data_path_test = ''

        self.classes = 50
        self.model_path_final = 'model1-1.pt'


        