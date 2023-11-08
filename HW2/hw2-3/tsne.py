import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

import model
from dataloader import MyDataloader
from constant import CONSTANT


def inference(encoder, classifier, source, target, df, do_tsne = True):
    all_emb = []
    all_st = []
    all_y = []
    
    for loader in [source, target]:
        all_pred = []
        for x, y in loader:
            x,y = x.to(C.device), y.to(C.device)
            emb = encoder(x)
            pred = classifier(emb)
            all_emb.append(emb.detach().cpu())
            all_st.append(torch.zeros(emb.shape[0]) if loader == source else torch.ones(emb.shape[0]))
            all_y.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())

    all_emb = torch.cat(all_emb, dim=0)
    all_st = torch.cat(all_st, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_pred = torch.argmax(torch.cat(all_pred, dim=0), dim=1)

    if do_tsne:
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000, n_jobs=-1)
        dann_tsne = tsne.fit_transform(all_emb.detach().cpu().numpy())

        plt.figure(figsize=(10, 10))
        plt.title('TSNE (by label)')
        s = plt.scatter(dann_tsne[:, 0], dann_tsne[:, 1], c=all_y, label=list(range(10)), cmap='gist_rainbow', s=5)
        plt.legend(handles=s.legend_elements()[0], labels=list(range(10)))
        plt.savefig(f'saved_plot/tsne_by_label_{C.source}_{C.target}.png')

        color_map = ListedColormap(['r','b'])
        plt.figure(figsize=(10, 10))
        plt.title('TSNE (by domain)')
        s = plt.scatter(dann_tsne[:, 0], dann_tsne[:, 1], c=all_st, cmap=color_map, s=5)
        plt.legend(handles=s.legend_elements()[0], labels=['source','target'])
        plt.savefig(f'saved_plot/tsne_by_domain_{C.source}_{C.target}.png')


if __name__ == '__main__':    
    C = CONSTANT()

    source = MyDataloader(C.source)
    source.setup(['valid'])
    target = MyDataloader(C.target)
    target.setup(['valid'])
    
    with torch.no_grad():
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        with open(C.encoder_path,'rb') as f:
            encoder.load_state_dict(torch.load(f))
        with open(C.classfier_path,'rb') as f:
            classifier.load_state_dict(torch.load(f))
        inference(encoder, classifier, source.loader['valid'], target.loader['valid'], target.df['valid'])