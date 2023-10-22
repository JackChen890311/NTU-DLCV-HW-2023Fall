import os
import time
import torch
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from model import MyModel
from dataloader import MyDataloader
from constant import CONSTANT

C = CONSTANT()

def train(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    accu, cnt = 0, 0
    for x,y in loader:
        x,y = x.to(C.device),y.to(C.device)
        optimizer.zero_grad()
        yhat = model(x)
        loss = loss_fn(yhat, y)
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
        loss = loss_fn(yhat, y)
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
    if 'scatter' in config:
        colors = plt.cm.rainbow(np.linspace(0, 1, C.classes))
        colors = [colors[int(i)] for i in config['scatter'][2]]
        plt.scatter(config['scatter'][0], config['scatter'][1], c=colors, s=3)
    if 'legend' not in config or config['legend']:
        plt.legend()
    plt.savefig(config['savefig'])
    plt.clf()


def dimensionReduction(model, loader, pca, tsne, e, start_time):
    model.eval()
    latent = torch.zeros((0, 1000))
    label = torch.zeros((0))
    for x,y in loader:
        x,y = x.to(C.device),y.to(C.device)
        yhat = model.backbone(x).cpu().detach()
        latent = torch.cat([latent, yhat])
        label = torch.cat([label, y.cpu()])
    pca_vec = pca.fit_transform(latent)
    tsne_vec = tsne.fit_transform(latent)
    label = label.numpy()

    config = {
        'title':'pca at epoch %d'%e,
        'xlabel':'x1',
        'ylabel':'x2',
        'data': [],
        'scatter': [pca_vec[:,0], pca_vec[:,1], label],
        'legend': False,
        'savefig':'output/%s/dr/pca_%d.png'%(start_time, e)
    }
    myplot(config)

    config = {
        'title':'tsne at epoch %d'%e,
        'xlabel':'x1',
        'ylabel':'x2',
        'data': [],
        'scatter': [tsne_vec[:,0], tsne_vec[:,1], label],
        'legend': False,
        'savefig':'output/%s/dr/tsne_%d.png'%(start_time, e)
    }
    myplot(config)


def test_model(path):
    model = MyModel()
    model.load_state_dict(torch.load(path))
    model = model.to(C.device)
    model.eval()
    dataloaders = MyDataloader()
    dataloaders.setup(['test'])
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, accu = test(model, dataloaders.loader['test'], loss_fn)
    return round(accu,4)


def main():
    model = MyModel()
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup(['train', 'valid'])
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=C.lr, weight_decay=C.wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=C.milestones, gamma=C.gamma)
    
    pca = PCA(n_components=2)
    tsne = TSNE(n_components=2, init='random', learning_rate='auto')

    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    if not os.path.exists('output'):
        os.mkdir('output')
    os.mkdir('output/%s'%start_time)
    os.mkdir('output/%s/dr'%start_time)

    with open('output/%s/log.txt'%start_time, 'w') as f:
        for key, value in vars(C).items():
            f.write(f"{key}: {value}\n")
        f.write('Optimizer:'+str(optimizer)+'\n')
        f.write('Scheduler:'+str(scheduler)+'\n')
        f.write('\n===== Model Structure =====')
        f.write('\n'+str(model)+'\n')
        f.write('===== Begin Training... =====\n')

    train_losses = []
    valid_losses = []
    train_accus = []
    valid_accus = []

    p_cnt = 0
    best_valid_loss = 1e10
    best_valid_accu = 0

    for e in tqdm(range(1,1+C.epochs)):
        train_loss, train_accu = train(model, dataloaders.loader['train'], optimizer, loss_fn)
        valid_loss, valid_accu = test(model, dataloaders.loader['valid'], loss_fn)

        scheduler.step()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accus.append(train_accu)
        valid_accus.append(valid_accu)
        print(f'Epoch = {e}, Train / Valid Loss = {round(train_loss, 6)} / {round(valid_loss, 6)}, ACCU = {round(train_accu, 4)} / {round(valid_accu, 4)}')
        
        if e % C.verbose == 0:
            x_axis = list(range(e))
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

        dimensionReduction(model, dataloaders.loader['valid'], pca, tsne, e, start_time)

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

        with open('output/%s/losses.pickle'%start_time, 'wb') as file:
            d = {'loss': [train_losses, valid_losses, best_valid_loss], 'accu' : [train_losses, valid_losses, best_valid_loss]}
            pk.dump(d, file)
        with open('output/%s/log.txt'%start_time, 'a') as f:
            f.write(f"Epoch {e}, Best valid loss: {round(best_valid_loss, 6)}, Best valid accuracy: {round(best_valid_accu, 4)}\n")
        
    print(f'Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}, Best valid accuracy: {round(best_valid_accu, 4)}')
    with open('output/%s/log.txt'%start_time, 'a') as f:
        f.write(f"Ending at epoch {e}. Best valid loss: {round(best_valid_loss, 6)}, Best valid accuracy: {round(best_valid_accu, 4)}\n")
        

if __name__ == '__main__':
    main()
    # print(test_model('output/2023-09-23~07:46:18/model.pt'))

