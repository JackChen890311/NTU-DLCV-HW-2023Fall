from torch import device

class CONSTANT():
    def __init__(self):
        self.epochs = 500
        self.lr = 5e-4
        self.wd = 1e-3
        self.bs = 16
        self.nw = 4
        self.pm = True
        self.milestones = [30, 100, 200]
        self.gamma = 0.5
        self.patience = 300
        self.verbose = 1
        self.device = device('cuda:0')

        # self.data_path_train_ssl = '../hw1_data/p2_data/mini/train'
        # self.data_path_train = '../hw1_data/p2_data/office/train'
        # self.data_path_valid = '../hw1_data/p2_data/office/val'
        # self.data_path_test = ''
        self.data_path_train_ssl = ''
        self.data_path_train = ''
        self.data_path_valid = ''
        self.data_path_test = ''

        self.classes = 65

        ''' SSL '''
        self.use_ssl = True
        self.train_ssl = False
        self.fix_backbone = False
        # self.model_path_ssl = 'output/2023-10-15~02:51:14/ssl/model_ssl_500.pt'
        # self.model_path_ssl = '../hw1_data/p2_data/pretrain_model_SL.pt'
        self.model_path_ssl = ''
        # if train_ssl == True, model_path_ssl will be ignored
        self.epochs_ssl = 3000
        self.lr_ssl = 3e-4
        self.wd_ssl = 1e-6
        self.patience_ssl = 3000
        self.save_freq_ssl = 20

        self.model_path_final = 'model1-2.pt'



        