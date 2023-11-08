import torch


class CONSTANT():
    def __init__(self, source = 'mnistm', target = 'usps'):
        self.epochs = 20
        self.lr = 5e-3
        self.mom = 0.9
        self.bs = 16
        self.nw = 16
        self.pm = True
        self.verbose = 5
        self.loss_coef = 0.5 # higher -> more domain loss
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'

        self.source = source
        self.target = target

        # self.mnistm_path = '../hw2_data/digits/mnistm/data/'
        # self.svhn_path = '../hw2_data/digits/svhn/data/'
        # self.usps_path = '../hw2_data/digits/usps/data/'
        # self.encoder_path = 'trained_models/encoder_dann_DANN_'+self.source+'_'+self.target+'.pt'
        # self.classfier_path = 'trained_models/classifier_dann_DANN_'+self.source+'_'+self.target+'.pt'

        self.mnistm_path = ''
        self.svhn_path = ''
        self.usps_path = ''
        self.encoder_path = ''
        self.classfier_path = ''


        self.final_encoder_path = 'models/encoder_dann_DANN_'+self.source+'_'+self.target+'.pt'
        self.final_classfier_path = 'models/classifier_dann_DANN_'+self.source+'_'+self.target+'.pt'