import torch


class CONSTANT():
    def __init__(self):
        self.epochs = 10
        self.lr = 1e-4
        self.wd = 1e-5
        self.bs_train = 64
        self.bs_valid = 64
        # TODO decrease batch size inference
        self.bs_infer = 32
        self.accu_step = 1
        self.bs_true = self.bs_train * self.accu_step
        self.nw = 4
        self.pm = True
        self.patience = 15
        self.verbose = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.PEFT_mode = 'lora' # none, adapter, prefix, lora
        self.adapter_size = 64
        self.prefix_cnt = 32
        self.lora_rank = 32
        
        self.max_seqlen = 128 if self.PEFT_mode != 'prefix' else 128 + self.prefix_cnt
        self.end_token_id = 50256
        self.beam_size = 5
        self.infer_step_greedy = 50
        self.infer_step_beam = 30

        # self.data_path_train = '../hw3_data/p2_data/images/train'
        # self.data_path_valid = '../hw3_data/p2_data/images/val'
        # self.data_path_test = '../hw3_data/p2_data/images/val'

        # self.json_path_train = '../hw3_data/p2_data/train.json'
        # self.json_path_valid = '../hw3_data/p2_data/val.json'
        # self.json_path_test = '../hw3_data/p2_data/val.json'

        # self.decoder_path = '../hw3_data/p2_data/decoder_model.bin'
        self.encoder_model_name = 'vit_large_patch14_clip_224'
        

        self.data_path_train = ''
        self.data_path_valid = ''
        self.data_path_test = ''

        self.json_path_train = ''
        self.json_path_valid = ''
        self.json_path_test = ''

        self.decoder_path = ''

        