from torch import device

class CONSTANT():
    def __init__(self):
        self.epochs = 100
        self.lr = 1e-4
        self.wd = 1e-4
        self.bs = 1
        self.nw = 16
        self.pm = True
        self.milestones = [10,50,90]
        self.gamma = 0.5
        self.patience = 30
        self.verbose = 1
        self.device = device('cuda:0')

        self.data_path_train = '../hw1_data/p3_data/train'
        self.data_path_valid = '../hw1_data/p3_data/validation'
        self.data_path_test = ''

        self.classes = 7
        self.output_train = 'output/masks/train'
        self.output_valid = 'output/masks/valid'
        self.output_test = 'output/masks/test'

        self.model_path_final = 'model1-3.pt'



        