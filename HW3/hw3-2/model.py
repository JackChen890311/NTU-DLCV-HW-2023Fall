import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from constant import CONSTANT
from tokenizer import BPETokenizer
from dataloader import MyDataloader
from decoder import Decoder, Config

C = CONSTANT()

class MyModel(nn.Module):
    # Implement your model here
    def __init__(self, encoder_name = C.encoder_model_name, decoder_path = C.decoder_path, save_map = False):
        # Initialize your model object
        super(MyModel, self).__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=True)
        cfg = Config(checkpoint=decoder_path, save_map=save_map)
        self.decoder = Decoder(cfg)
        self.encoder_proj = nn.Linear(1024, cfg.n_embd)

    def forward(self, image, text_id):
        # Return the output of model given the input x
        image_feat = self.encoder.forward_features(image)
        image_emb = self.encoder_proj(image_feat)
        text_pred = self.decoder(text_id, image_emb)
        return text_pred

    # TODO add auto stop when seeing 50256
    def autoreg_infer(self, image, step = 100):
        # Implement autoregressive inference here
        image_feat = self.encoder.forward_features(image)
        image_emb = self.encoder_proj(image_feat)
        text_id = torch.ones((image_emb.size(0), 1), dtype=torch.long, device=C.device) * 50256

        if C.PEFT_mode == 'prefix':
            prefix = torch.arange(C.prefix_cnt).unsqueeze(0).repeat(image_emb.size(0),1).to(C.device)
            text_id = torch.cat((prefix, text_id), dim=-1)

        for i in range(step):
            text_pred = self.decoder(text_id, image_emb)
            if C.PEFT_mode == 'prefix':
                text_pred = text_pred[:, C.prefix_cnt:]
            text_pred_id = torch.argmax(text_pred, dim=-1)
            text_id = torch.cat((text_id, text_pred_id[:,i].unsqueeze(1)), dim=-1)
            
        return text_id

    def autoreg_infer_beam(self, image, beam, step = 100):
        # Implement autogressive inference with beam search here
        image_feat = self.encoder.forward_features(image)
        image_emb = self.encoder_proj(image_feat) # batch x (patches + 1) x 768 (patches = 196, 256...)
        batch_cnt = image_emb.size(0)
        text_id = torch.ones((batch_cnt, 1, 1), dtype=torch.long, device=C.device) * 50256 # batch x context(1) x path
        text_prob = torch.ones((batch_cnt, 1), dtype=torch.float) # batch x path

        if C.PEFT_mode == 'prefix':
            prefix = torch.arange(C.prefix_cnt).unsqueeze(0).unsqueeze(2).repeat(image_emb.size(0),1,1).to(C.device)
            text_id = torch.cat((prefix, text_id), dim=1)

        for i in range(step):
            text_pred_all = torch.zeros((batch_cnt, 50257, 0))
            path_cnt = text_id.size(2)
            for j in range(path_cnt):
                text_pred_path = F.softmax(self.decoder(text_id[:, :, j], image_emb), dim=-1).detach().cpu() # batch x 1 x 50257
                if C.PEFT_mode == 'prefix':
                    text_pred_path = text_pred_path[:, C.prefix_cnt:]
                text_pred_all = torch.cat((text_pred_all, text_pred_path[:, i, :].unsqueeze(2)), dim=-1)
            text_pred_topk = torch.topk(text_pred_all, beam, dim=1)
            text_pred_id = text_pred_topk.indices # batch x beam x path
            text_pred_scores = text_pred_topk.values # batch x beam x path

            text_prob_prev = text_prob.unsqueeze(1).repeat(1, beam, 1) # batch x beam x path
            text_pred_scores = text_prob_prev * text_pred_scores * 2 # batch x beam x path

            new_text_id = text_id.repeat(1, 1, beam).detach().cpu() # batch x context x beam*path
            new_text_pred_id = text_pred_id.view(batch_cnt, 1, beam*path_cnt) # batch x 1 x beam*path
            new_text_pred_scores = text_pred_scores.view(batch_cnt, 1, beam*path_cnt) # batch x 1 x beam*path
            final = torch.cat((new_text_id, new_text_pred_id), dim=1) # batch x context+2 x beam*path
            
            # Sort by probability
            text_prob = torch.zeros((batch_cnt, beam*path_cnt), dtype=torch.float)
            for k in range(batch_cnt):
                sort_score, indexes = new_text_pred_scores[k].squeeze().sort(descending=True)
                final[k] = final[k][:, indexes]
                text_prob[k] = sort_score
            text_id = final.to(C.device)
            text_id = text_id[:, :, :beam]
            text_prob = text_prob[:, :beam]

        return text_id[:,:,0]


if __name__ == '__main__':
    C = CONSTANT()
    model = MyModel().to(C.device)
    # print(model)
    
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])

    for img, txt, txt_decode in dataloaders.loader['test']:
        img, txt, txt_decode = img.to(C.device), txt.to(C.device), txt_decode.to(C.device)
        txt_pred = model(img, txt)
        print(img.shape,txt.shape)
        print(model.encoder_proj(model.encoder.forward_features(img)).shape)
        print(txt_pred.shape)
        print(txt_decode.shape)

        txt_pred_id = torch.argmax(txt_pred, dim=-1)
        tokenizer = BPETokenizer('encoder.json', 'vocab.bpe')

        txt_decode = txt_decode[0].detach().cpu().tolist()
        txt_pred_id = txt_pred_id[0].detach().cpu().tolist()

        print(txt_pred_id[:15])
        print(txt_decode[:15])  

        # if C.PEFT_mode == 'prefix':
        #     txt_pred_id = txt_pred_id[C.prefix_cnt:]
        #     txt_decode = txt_decode[C.prefix_cnt:]

        #     txt_pred_id = txt_pred_id[1:txt_pred_id[1:].index(50256)]
        #     txt_decode = txt_decode[:txt_decode.index(50256)]
        
        # print(txt_pred_id[:15])
        # print(txt_decode[:15])        

        label = tokenizer.decode(txt_decode)
        predict = tokenizer.decode(txt_pred_id)
        print('=====')
        print('Truth:',label)
        print('Prediction:',predict)
        # break