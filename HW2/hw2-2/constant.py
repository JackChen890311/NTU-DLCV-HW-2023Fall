import torch

class CONSTANT():
    def __init__(self):

        self.num_timesteps = 1000
        self.timesteps = 50
        self.eta = 0
        self.beta_start = 1e-4
        self.beta_end = 2e-2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.model_path = '../hw2_data/face/UNet.pt'
        # self.gt_path = '../hw2_data/face/GT/'
        # self.noise_path = '../hw2_data/face/noise/'
        # self.out_path = './output/'

        self.model_path = ''
        self.gt_path = ''
        self.noise_path = ''
        self.out_path = ''
