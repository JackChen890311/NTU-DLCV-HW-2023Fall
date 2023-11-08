import torch
import train
import model
from dataloader import AllDataloader
from constant import CONSTANT

C = CONSTANT()

def main():
    adl = AllDataloader()
    adl.setup(C.source, C.target, ['train', 'valid'])

    # Source only and DANN
    train_loaders = [adl.source_loader.loader['train'], adl.target_loader.loader['train']]
    valid_loaders = [adl.source_loader.loader['valid'], adl.target_loader.loader['valid']]

    # Source = Target
    train_loaders_st = [adl.target_loader.loader['train'], adl.target_loader.loader['train']]
    valid_loaders_st = [adl.target_loader.loader['valid'], adl.target_loader.loader['valid']]


    if torch.cuda.is_available():
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()
        discriminator = model.Discriminator().cuda()

        so_accu = train.source_only(encoder, classifier, train_loaders, valid_loaders, 'SO_'+C.source+'_'+C.target)
        dann_accu = train.dann(encoder, classifier, discriminator, train_loaders, valid_loaders, 'DANN_'+C.source+'_'+C.target)
        st_accu = train.source_only(encoder, classifier, train_loaders_st, valid_loaders_st, 'ST_'+C.source+'_'+C.target)
        print('===== End of experiments =====')
        print('Source Only: {:.6f}\tDANN: {:.6f}\tSource-Target: {:.6f}'.format(so_accu, dann_accu, st_accu))
    else:
        print("There is no GPU -_-!")


if __name__ == "__main__":
    main()