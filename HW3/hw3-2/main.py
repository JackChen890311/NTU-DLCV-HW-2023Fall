import os
import time
import json
import torch
import random
import pickle as pk
import loralib as lora 
from tqdm import tqdm
from matplotlib import pyplot as plt

from constant import CONSTANT
from model import MyModel
from dataloader import MyDataloader
from tokenizer import BPETokenizer
from p2_evaluate import CIDERScore, CLIPScore, getGTCaptions
from inference import TestDataloader, run_inference

C = CONSTANT()
tokenizer = BPETokenizer('encoder.json', 'vocab.bpe')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

amp_enable = False
amp_bf16 = True
amp_dtype = torch.bfloat16 if amp_bf16 else torch.float16
amp_device = 'cuda' if torch.cuda.is_available() else 'cpu'


def show_demo_cases(yhat, txt_decode):
    try:
        idx = random.randint(0, yhat.size(0)-1)
        if C.PEFT_mode == 'prefix':
            yhat = yhat[:, C.prefix_cnt:]
            txt_decode = txt_decode[:, C.prefix_cnt:]

        # label
        y_id = txt_decode[idx].detach().cpu().tolist()
        y_id = y_id[:y_id.index(-100)-1]

        # predict
        yhat_id = torch.argmax(yhat, dim=-1)
        yhat_id = yhat_id[idx].detach().cpu().tolist()
        try:
            yhat_endid = yhat_id.index(50256)
        except:
            yhat_endid = len(yhat_id)
        yhat_id = yhat_id[:yhat_endid]
       
        label = tokenizer.decode(y_id)
        predict = tokenizer.decode(yhat_id)
        print('=====')
        print('Truth:',label)
        print('Prediction:',predict)
    except Exception as e:
        print(e)


def train(model, loader, optimizer, scaler, scheduler, loss_fn):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for idx, (img,txt,txt_decode) in enumerate(tqdm(loader)):
        img,txt,txt_decode = img.to(C.device),txt.to(C.device),txt_decode.to(C.device)

        with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enable):
            yhat = model(img,txt)
    
        # show_demo_cases(yhat,txt_decode)
        yhat = yhat.permute(0,2,1)
        loss = loss_fn(yhat,txt_decode)
        loss /= C.accu_step 
        
        scaler.scale(loss).backward()
        if idx % C.accu_step == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * C.accu_step
        # print('Loss:',loss.item() * C.accu_step)

        scheduler.step()

    if len(loader) % C.accu_step != 0:
        optimizer.zero_grad(set_to_none=True)
        scaler.step(optimizer)
        scaler.update()

    return total_loss/len(loader)


def valid(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img,txt,txt_decode in tqdm(loader):
            img,txt,txt_decode = img.to(C.device),txt.to(C.device),txt_decode.to(C.device)
            with torch.autocast(device_type=amp_device, dtype=amp_dtype, enabled=amp_enable):
                yhat = model(img,txt)
            # show_demo_cases(yhat,txt_decode)
            yhat = yhat.permute(0,2,1)
            loss = loss_fn(yhat,txt_decode)
            total_loss += loss.item() 
    return total_loss/len(loader)


def test(model, loader, filename, json_path, img_root):
    model.eval()
    result = run_inference(model, loader, filename, 'greedy')

    with open(json_path, 'r') as f:
        annotations = json.load(f)
    gts = getGTCaptions(annotations)

    assert type(result) is dict
    assert set(result.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in result.values()])
    
    cider_score = CIDERScore()(result, gts)
    clip_score = CLIPScore()(result, img_root)

    return cider_score, clip_score


def myplot(config):
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    for label in config['data']:
        plt.plot(config['data'][label][0], config['data'][label][1], label=label)
    plt.legend()
    plt.savefig(config['savefig'])
    plt.clf()


def test_model(modelPath):
    # Model
    model = MyModel()
    model.load_state_dict(torch.load(os.path.join(modelPath, 'model.pt')), strict=False)
    if C.PEFT_mode == 'lora':
        model.load_state_dict(torch.load(os.path.join(modelPath,'lora.pt')), strict=False)
    model = model.to(C.device)
    model.eval()

    # Data
    dataloaders = MyDataloader()
    dataloaders.setup(['valid'])

    # Valid
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    valid_loss = valid(model, dataloaders.loader['valid'], loss_fn)
    print('Validation Loss:', valid_loss)

    # Add your own metrics here
    testloader = TestDataloader(C.data_path_valid)
    testloader.setup(['test'])
    cider_score, clip_score = test(model, testloader.loader['test'], testloader.filename['test'], C.json_path_valid, C.data_path_valid)
    print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")


def main():
    # Load model and data
    model = MyModel()
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'valid', 'test'])

    testloader = TestDataloader(C.data_path_valid)
    testloader.setup(['test'])

    # You can adjust these as your need
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloaders.loader['train'])*C.epochs)#milestones=C.milestones, gamma=C.gamma)

    # Speed up
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)

    # PEFT setting
    if C.PEFT_mode == 'none':
        pass
    elif C.PEFT_mode == 'adapter' or C.PEFT_mode == 'prefix':
        for name, param in model.named_parameters():
            if C.PEFT_mode not in name and 'crossattn' not in name and 'encoder_proj' not in name:
                param.requires_grad = False
    elif C.PEFT_mode == 'lora':
        lora.mark_only_lora_as_trainable(model)
        for name, param in model.named_parameters():
            if 'crossattn' in name or 'encoder_proj' in name:
                param.requires_grad = True
    
    num_para_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('PEFT:', C.PEFT_mode, ', Total Paramters:', num_para_grad)

    # Set up output directory
    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    if not os.path.exists('output'):
        os.mkdir('output')
    os.mkdir('output/%s'%start_time)
    with open('output/%s/log.txt'%start_time, 'w') as f:
        for key, value in vars(C).items():
            f.write(f"{key}: {value}\n")
        f.write('Optimizer:'+str(optimizer)+'\n')
        f.write('Scheduler:'+str(scheduler)+'\n')
        f.write('Loss Function:'+str(loss_fn)+'\n')
        f.write('PEFT: '+C.PEFT_mode+', Total Paramters:'+str(num_para_grad)+'\n')
        f.write('\n===== Model Structure =====')
        f.write('\n'+str(model)+'\n')
        f.write('===== Begin Training... =====\n')

    # Start training
    train_losses = []
    valid_losses = []
    p_cnt = 0
    best_valid_loss = 1e10
    best_cider_score = 0
    best_clip_score = 0

    for e in tqdm(range(1,1+C.epochs)):
        # Train and valid step
        train_loss = train(model, dataloaders.loader['train'], optimizer, scaler, scheduler, loss_fn)
        valid_loss = valid(model, dataloaders.loader['valid'], loss_fn)
        cider_score, clip_score = test(model, testloader.loader['test'], testloader.filename['test'], C.json_path_valid, C.data_path_valid)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}')
        print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")
        
        # Plot loss
        if e % C.verbose == 0:
            x_axis = list(range(1,1+e))
            config = {
                'title':'Loss',
                'xlabel':'Epochs',
                'ylabel':'Loss',
                'data':{
                    'Train':[x_axis, train_losses],
                    'Valid':[x_axis, valid_losses]
                },
                'savefig':'output/%s/loss.png'%start_time
            }
            myplot(config)

        # Save best model and early stopping
        trainable_weights = [name for name, param in model.named_parameters() if param.requires_grad]
        save_weights = {k: v for k, v in model.state_dict().items() if k in trainable_weights}
        print('Total Paramters:', sum(p.numel() for p in save_weights.values()))

        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            torch.save(save_weights, 'output/%s/trainable_weights.pt'%start_time)
        else:
            p_cnt += 1
            if p_cnt == C.patience:
                print('Early Stopping at epoch', e)
                break
        
        if cider_score > best_cider_score:
            best_cider_score = cider_score
            torch.save(save_weights, 'output/%s/trainable_weights_cider.pt'%start_time)
        if clip_score > best_clip_score:
            best_clip_score = clip_score
            torch.save(save_weights, 'output/%s/trainable_weights_clip.pt'%start_time)
        
        # Write log
        with open('output/%s/log.txt'%start_time, 'a') as f:
            f.write(f"Epoch {e}, Best valid loss: {round(best_valid_loss, 6)}\n")
            f.write(f"CIDEr: {cider_score} | CLIPScore: {clip_score}\n")
            
        with open('output/%s/losses.pickle'%start_time, 'wb') as file:
            pk.dump([train_losses, valid_losses, best_valid_loss], file)
    
    # Ends training
    print(f'Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}')
    with open('output/%s/log.txt'%start_time, 'a') as f:
        f.write(f"Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}\n")



if __name__ == "__main__":
    main()
    # test_model('output/2023-11-22~02:46:51/')
