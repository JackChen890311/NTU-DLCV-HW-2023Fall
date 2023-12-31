import os
import math
import torch
import collections
import pickle as pk
import loralib as lora
from torch import nn, Tensor
import torch.nn.functional as F

from constant import CONSTANT

CONST = CONSTANT()

class Config:

    def __init__(self, checkpoint=None, save_map=False):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.save_map = save_map

class Attention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd) if CONST.PEFT_mode != 'lora' else lora.Linear(cfg.n_embd, 3 * cfg.n_embd, r=CONST.lora_rank)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd) if CONST.PEFT_mode != 'lora' else lora.Linear(cfg.n_embd, cfg.n_embd, r=CONST.lora_rank)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        size = cfg.block_size
        self.register_buffer('bias', torch.tril(torch.ones(size, size)).view(1, 1, size, size))
        
        if CONST.PEFT_mode == 'adapter':
            self.adapter_c_attn = nn.Sequential(collections.OrderedDict([
                ('c_fc', nn.Linear(cfg.n_embd, CONST.adapter_size)),
                ('drop_fc', nn.Dropout(0.25)),
                ('act', nn.Tanh()),
                ('c_proj', nn.Linear(CONST.adapter_size, cfg.n_embd)),
                ('drop_proj', nn.Dropout(0.25)),
            ]))

    def forward(self, x):
        B, T, C = x.size() # batch, context, embedding
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        result = self.c_proj(att)

        if CONST.PEFT_mode == 'adapter':
            result = result + self.adapter_c_attn(result)
        return result
    
class CrossAttention(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.c_attn2 = nn.Linear(cfg.n_embd, 2 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = 1
        self.n_embd = cfg.n_embd

    def forward(self, text_emb, img_emb, save_map=False):
        B, T, C = text_emb.size() # batch, context, embedding
        B2, T2, C2 = img_emb.size() # batch, context, embedding
        q = self.c_attn(text_emb)
        k, v = self.c_attn2(img_emb).split(self.n_embd, dim=2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B2, T2, self.n_head, C2 // self.n_head).transpose(1, 2)
        v = v.view(B2, T2, self.n_head, C2 // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        if save_map:
            new_att = att.detach().clone()
            if os.path.exists('attn_map.pkl'):
                with open('attn_map.pkl', 'rb') as f:
                    old_att = pk.load(f)
                try:
                    new_att += old_att
                except:
                    pass
            with open('attn_map.pkl', 'wb') as f:
                pk.dump(new_att, f)
        result = self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))
        return result

class Block(nn.Module):

    def __init__(self, cfg, layer):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(collections.OrderedDict([
            ('c_fc', nn.Linear(cfg.n_embd, 4 * cfg.n_embd) if CONST.PEFT_mode != 'lora' else lora.Linear(cfg.n_embd, 4 * cfg.n_embd, r=CONST.lora_rank)),
            ('act', nn.GELU(approximate='tanh')),
            ('c_proj', nn.Linear(4 * cfg.n_embd, cfg.n_embd) if CONST.PEFT_mode != 'lora' else lora.Linear(4 * cfg.n_embd, cfg.n_embd, r=CONST.lora_rank))
        ]))
        self.ln_3 = nn.LayerNorm(cfg.n_embd)
        self.crossattn = CrossAttention(cfg)
        self.block_size = cfg.block_size
        self.cfg = cfg
        self.layer = layer

        if CONST.PEFT_mode == 'adapter':
            self.adapter_mlp = nn.Sequential(collections.OrderedDict([
                ('c_fc', nn.Linear(cfg.n_embd, CONST.adapter_size)),
                ('drop_fc', nn.Dropout(0.25)),
                ('act', nn.Tanh()),
                ('c_proj', nn.Linear(CONST.adapter_size, cfg.n_embd)),
                ('drop_proj', nn.Dropout(0.25)),
            ]))

    def forward(self, input_tensor):
        (text_emb, img_emb) = input_tensor
        text_emb = text_emb + self.attn(self.ln_1(text_emb))
        if self.cfg.save_map and self.layer <= 8 and self.layer >= 3:
            text_emb = text_emb + self.crossattn(self.ln_3(text_emb), img_emb, True)
        else:
            text_emb = text_emb + self.crossattn(self.ln_3(text_emb), img_emb)

        if CONST.PEFT_mode == 'adapter':
            ada_emb = self.mlp(self.ln_2(text_emb))
            text_emb = text_emb + self.adapter_mlp(ada_emb) + ada_emb
        else:
            text_emb = text_emb + self.mlp(self.ln_2(text_emb))
        return (text_emb, img_emb)

class Decoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe = nn.Embedding(cfg.block_size, cfg.n_embd),
            h = nn.Sequential(*[Block(cfg, layer) for layer in range(cfg.n_layer)]),
            ln_f = nn.LayerNorm(cfg.n_embd)
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        # load checkpoint
        if self.cfg.checkpoint is not None:
            print('Loading decoder checkpoint...')
            state_dict = torch.load(self.cfg.checkpoint)
            transposed = [ '.c_attn.weight', '.c_fc.weight', '.c_proj.weight' ]
            for key, value in state_dict.items():
                if any(key.endswith(w) for w in transposed):
                    state_dict[key] = value.t()
            self.transformer.load_state_dict(state_dict, strict=False)

        if CONST.PEFT_mode == 'prefix':
            self.prefix_proj = nn.Embedding(CONST.prefix_cnt, cfg.n_embd)

    def forward(self, text_id: Tensor, image_emb: Tensor):
        image_emb = image_emb.float()
        text_id = torch.narrow(text_id, 1, 0, min(text_id.size(1), self.block_size))
        pos = torch.arange(text_id.size()[1], dtype=torch.long, device=text_id.device).unsqueeze(0)
        
        if CONST.PEFT_mode == 'prefix':
            prefix_emb = self.prefix_proj(text_id[:, :CONST.prefix_cnt])
            text_emb = self.transformer.wte(text_id[:, CONST.prefix_cnt:])
            text_emb = torch.cat((prefix_emb, text_emb), dim=1) + self.transformer.wpe(pos)#[:, :-CONST.prefix_cnt])
        else:
            text_emb = self.transformer.wte(text_id) + self.transformer.wpe(pos)
        (text_emb, image_emb) = self.transformer.h((text_emb, image_emb))
        result = self.lm_head(self.transformer.ln_f(text_emb))
        return result
