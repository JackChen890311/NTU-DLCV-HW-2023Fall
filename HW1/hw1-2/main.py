import os
import time
import torch
import pickle as pk
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import MyModel
from dataloader import MyDataloader
from constant import CONSTANT
from byol_pytorch import BYOL

C = CONSTANT()

def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    accu, cnt = 0, 0
    for x,y in loader:
        optimizer.zero_grad()
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # accu
        y_idx = torch.argmax(yhat, 1)
        accu += torch.sum(y_idx == y).item()
        cnt += len(y)
    return total_loss/len(loader), accu/cnt


def test(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    accu, cnt = 0, 0
    for x,y in loader:
        x,y = x.to(C.device),y.to(C.device)
        yhat = model(x)
        loss = loss_fn(yhat,y)
        total_loss += loss.item()
        # accu
        y_idx = torch.argmax(yhat, 1)
        accu += torch.sum(y_idx == y).item()
        cnt += len(y)
    return total_loss/len(loader), accu/cnt


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
    loss, accu = test(model, dataloaders.loader['test'], loss_fn)
    return round(accu,4)


def sslPretrain(model, train_loader, valid_loader, start_time):
    # Reference: https://github.com/lucidrains/byol-pytorch
    learner = BYOL(
        model,
        image_size = 128,
        hidden_layer = 'avgpool',
    )
    optimizer = torch.optim.Adam(learner.parameters(), lr=C.lr_ssl, weight_decay=C.wd_ssl)
    os.mkdir('output/%s/ssl'%start_time)
    
    train_losses = []
    valid_losses = []
    train_loss, valid_loss = 0, 0
    best_valid_loss = 1e10
    
    for e in tqdm(range(1,1+C.epochs_ssl)):
        # Train
        model.train()
        for x,y in train_loader:
            x = x.to(C.device)
            loss = learner(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
            train_loss += loss.item()
        
        # Valid
        model.eval()
        for x,y in valid_loader:
            x = x.to(C.device)
            loss = learner(x)
            valid_loss += loss.item()

        
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)


        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'output/%s/ssl/model_ssl.pt'%start_time)
        else:
            p_cnt += 1
            if p_cnt == C.patience_ssl:
                print('Early Stopping at epoch', e)
                break
        
        if e % C.save_freq_ssl == 0:
            torch.save(model.state_dict(), 'output/%s/ssl/model_ssl_%d.pt'%(start_time,e))

        if e % C.verbose == 0:
            x_axis = list(range(1,1+e))
            config = {
                    'title':'Loss',
                    'xlabel':'Epochs',
                    'ylabel':'Loss',
                    'data':{
                        'train':[x_axis, train_losses],
                        'valid':[x_axis, valid_losses],
                    },
                    'savefig':'output/%s/loss_ssl.png'%start_time
            }
            myplot(config)
            
        print(f'SSL Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}')
        with open('output/%s/log.txt'%start_time, 'a') as f:
            f.write(f"SSL Epoch {e}, Best Valid loss: {round(best_valid_loss, 6)}\n")
    
    with open('output/%s/log.txt'%start_time, 'a') as f:
            f.write(f"===== SSL end at epoch {e}, Best Valid loss: {round(best_valid_loss, 6)} =====\n")
    return 


def main():
    # Load model and data
    model = MyModel()
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'train_ssl', 'valid', 'valid_ssl'])

    # You can adjust these as your need
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.milestones, gamma=C.gamma)

    # Set up output directory
    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    if not os.path.exists('output'):
        os.mkdir('output')
    os.mkdir('output/%s'%start_time)
    with open('output/%s/log.txt'%start_time, 'w') as f:
        for key, value in vars(C).items():
            f.write(f"{key}: {value}\n")
        f.write('Optimizer: '+str(optimizer)+'\n')
        f.write('Scheduler: '+str(scheduler)+'\n')
        f.write('Preprocess: \n')
        for key, value in vars(dataloaders.P).items():
            f.write(f"{key}: {value}\n")
        f.write('\n===== Model Structure =====')
        f.write('\n'+str(model)+'\n')
        f.write('===== Begin Training... =====\n')

    # ssl pretraining
    if C.use_ssl:
        if C.train_ssl:
            sslPretrain(model.backbone, dataloaders.loader['train_ssl'], dataloaders.loader['valid_ssl'], start_time)
        model_path_ssl = 'output/%s/model_ssl.pt'%start_time if C.train_ssl else C.model_path_ssl

        with open('output/%s/log.txt'%start_time, 'a') as f:
            model.backbone.load_state_dict(torch.load(model_path_ssl))
            f.write('SSL model using from: ' + model_path_ssl+'\n')
    
    # Freeze backbone
    if C.fix_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    # Start training
    train_losses = []
    valid_losses = []
    train_accus = []
    valid_accus = []

    p_cnt = 0
    best_valid_loss = 1e10
    best_valid_accu = 0

    for e in tqdm(range(1,1+C.epochs)):
        # Train and valid step
        train_loss, train_accu = train(model, dataloaders.loader['train'], optimizer, loss_fn)
        valid_loss, valid_accu = test(model, dataloaders.loader['valid'], loss_fn)

        scheduler.step()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accus.append(train_accu)
        valid_accus.append(valid_accu)
        print(f'Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}, ACCU = {round(train_accu, 4)} / {round(valid_accu, 4)}')
        
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
                'title':'Accuracy',
                'xlabel':'Epochs',
                'ylabel':'Accuracy',
                'data':{
                    'Train':[x_axis, train_accus],
                    'Valid':[x_axis, valid_accus]
                },
                'savefig':'output/%s/accu.png'%start_time
            }
            myplot(config)

        # Save best model and early stopping
        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            best_valid_accu = valid_accu
            torch.save(model.state_dict(), 'output/%s/model.pt'%start_time)
        else:
            p_cnt += 1
            if p_cnt == C.patience:
                print('Early Stopping at epoch', e)
                break
        
        # Write log
        with open('output/%s/losses.pickle'%start_time, 'wb') as file:
            d = {'loss': [train_losses, valid_losses, best_valid_loss], 'accu' : [train_accus, valid_accus, best_valid_accu]}
            pk.dump(d, file)
        with open('output/%s/log.txt'%start_time, 'a') as f:
            f.write(f"Epoch {e}, Best valid loss: {round(best_valid_loss, 6)}, Best valid accuracy: {round(best_valid_accu, 4)}\n")
    
    # Ends training
    print(f'Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}, Best valid accuracy: {round(best_valid_accu, 4)}')
    with open('output/%s/log.txt'%start_time, 'a') as f:
        f.write(f"Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}, Best valid accuracy: {round(best_valid_accu, 4)}\n")


if __name__ == '__main__':
    main()

