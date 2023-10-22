import os
import time
import torch
import pickle as pk
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image

from model import MyModel
from dataloader import MyDataloader
from constant import CONSTANT
from mean_iou_evaluate import read_masks, mean_iou_score

C = CONSTANT()

def train(model, loader, optimizer, loss_fn, filenames, paths):
    model.train()
    total_loss = 0
    for x,y in loader:
        optimizer.zero_grad()
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss/len(loader)


def test(model, loader, loss_fn, filenames, paths):
    model.eval()
    total_loss = 0
    cnt = 0
    for x,y in loader:
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)
        total_loss += loss.item()
        # for miou
        yidx = torch.argmax(yhat, dim = 1)
        for i in range(y.shape[0]):
            rgbimg = idxtoRGB(yidx[i].squeeze())
            save_image(rgbimg, f'%s/%s_mask.png'%(paths[1], filenames[cnt]))
            cnt += 1

    pred = read_masks(paths[1])
    labels = read_masks(paths[0])
    miou = mean_iou_score(pred, labels)

    return total_loss/len(loader), miou


def idxtoRGB(idximg):
    rgbimg = torch.zeros((3,idximg.shape[0],idximg.shape[1]))
    idximg2 = torch.zeros((idximg.shape[0],idximg.shape[1]))

    idximg2[idximg == 0] = 3  # (Cyan: 011) Urban land 
    idximg2[idximg == 1] = 6  # (Yellow: 110) Agriculture land 
    idximg2[idximg == 2] = 5  # (Purple: 101) Rangeland 
    idximg2[idximg == 3] = 2  # (Green: 010) Forest land 
    idximg2[idximg == 4] = 1  # (Blue: 001) Water 
    idximg2[idximg == 5] = 7  # (White: 111) Barren land 
    idximg2[idximg == 6] = 0  # (Black: 000) Unknown 

    rgbimg[0] = idximg2 >= 4
    rgbimg[1] = (idximg2 % 4) >= 2
    rgbimg[2] = (idximg2 % 2) >= 1
    return rgbimg


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
    model.load_state_dict(torch.load(modelPath))
    model = model.to(C.device)
    model.eval()

    # Data
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])

    # Add your own metrics here
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, miou = test(model, dataloaders.loader['test'], loss_fn, dataloaders.filenames['test'], (C.data_path_test, C.output_test))
    return loss, miou


def main():
    # Load model and data
    model = MyModel()
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'valid', 'test'])

    # You can adjust these as your need
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.lr, weight_decay=C.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.milestones, gamma=C.gamma)

    # Set up output directory
    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    dirs = ['output', 'output/masks', 'output/masks/train', 'output/masks/valid', 'output/masks/test']
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)
    os.mkdir('output/%s'%start_time)
    with open('output/%s/log.txt'%start_time, 'w') as f:
        for key, value in vars(C).items():
            f.write(f"{key}: {value}\n")
        f.write('Optimizer:'+str(optimizer)+'\n')
        f.write('Scheduler:'+str(scheduler)+'\n')
        f.write('\n===== Model Structure =====')
        f.write('\n'+str(model)+'\n')
        f.write('===== Begin Training... =====\n')

    # Start training
    train_losses = []
    valid_losses = []
    valid_mious = []
    p_cnt = 0
    best_valid_loss = 1e10
    best_valid_miou = 0
    torch.save(model.state_dict(), 'output/%s/model_start.pt'%start_time)

    for e in tqdm(range(1,1+C.epochs)):
        # Train and valid step
        train_loss = train(model, dataloaders.loader['train'], optimizer, loss_fn, dataloaders.filenames['train'], (C.data_path_train, C.output_train))
        valid_loss, valid_miou = test(model, dataloaders.loader['valid'], loss_fn, dataloaders.filenames['valid'], (C.data_path_valid, C.output_valid))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_mious.append(valid_miou)
        print(f'Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}, Valid miou = {round(valid_miou, 4)}')
        scheduler.step()
        
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

            config = {
                'title':'mIoU',
                'xlabel':'Epochs',
                'ylabel':'mIoU',
                'data':{
                    'Valid':[x_axis, valid_mious]
                },
                'savefig':'output/%s/miou.png'%start_time
            }
            myplot(config)
            torch.save(model.state_dict(), 'output/%s/model_%d.pt'%(start_time,e))

        # Save best model and early stopping
        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            best_valid_miou = valid_miou
            torch.save(model.state_dict(), 'output/%s/model_end.pt'%start_time)
        else:
            p_cnt += 1
            if p_cnt == C.patience:
                print('Early Stopping at epoch', e)
                break
        
        # Write log
        with open('output/%s/log.txt'%start_time, 'a') as f:
            f.write(f"Epoch {e}: Best valid loss: {round(best_valid_loss, 6)}, Best valid miou = {round(best_valid_miou, 4)}\n")
        with open('output/%s/losses.pickle'%start_time, 'wb') as file:
            pk.dump([train_losses, valid_losses, best_valid_loss], file)
    
    # Ends training
    endText = f'Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}, Best valid miou = {round(best_valid_miou, 4)}'
    print(endText)
    with open('output/%s/log.txt'%start_time, 'a') as f:
        f.write(endText + '\n')


if __name__ == '__main__':
    main()
    # print(test_model('./output/2023-10-13~22:39:05/model_2.pt'))

