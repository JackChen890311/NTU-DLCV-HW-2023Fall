import torch

class CONSTANT():
    def __init__(self):
        self.epochs = 50
        self.lr = 1e-4
        self.bs = 256
        self.nw = 16
        self.pm = True
        self.verbose = 10
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

        self.n_T = 501 # 500
        self.n_feat = 128 # 128 ok, 256 better (but slower)
        self.n_classes = 10
        self.save_model = True

        self.w_infer = 5

        # self.data_path = '../hw2_data/digits/mnistm/data'
        # self.csv_path_train = '../hw2_data/digits/mnistm/train.csv'
        # self.csv_path_test = '../hw2_data/digits/mnistm/test.csv'
        # self.csv_path_val = '../hw2_data/digits/mnistm/val.csv'

        # self.infer_save_dir = './output/infer/'
        # self.infer_eval_dir = '../2-1_out/'
        # self.model_path = './output/2023-10-24~04:43:54/train/model_49.pth'

        self.data_path = ''
        self.csv_path_train = ''
        self.csv_path_test = ''
        self.csv_path_val = ''

        self.infer_save_dir = ''
        self.infer_eval_dir = ''
        self.model_path = ''