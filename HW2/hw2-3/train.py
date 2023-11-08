import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

import test
from utils import save_model, set_model_mode, optimizer_scheduler

from constant import CONSTANT

C = CONSTANT()


def myplot(config):
    plt.title(config['title'])
    plt.xlabel(config['xlabel'])
    plt.ylabel(config['ylabel'])
    for label in config['data']:
        plt.plot(config['data'][label][0], config['data'][label][1], label=label)
    plt.legend()
    plt.savefig(config['savefig'])
    plt.clf()


def source_only(encoder, classifier, train_loaders, valid_loaders, save_name):
    source_train_loader, target_train_loader = train_loaders
    source_test_loader, target_test_loader = valid_loaders
    print("Source-only training")
    classifier_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(classifier.parameters()),
        lr=C.lr, momentum=C.mom)
    
    total_losses = []
    ACCU = [[], []]
    bestaccu = 0
    bestloss = 1e6
    for epoch in range(C.epochs):
        total_loss = 0
        print('Epoch : {}'.format(epoch), end=' ')
        set_model_mode('train', [encoder, classifier])

        start_steps = epoch * len(source_train_loader)
        total_steps = C.epochs * len(target_train_loader)
        
        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):
            source_image, source_label = source_data
            p = float(batch_idx + start_steps) / total_steps

            # source_image = torch.cat((source_image, source_image, source_image), 1)  # MNIST convert to 3 channel
            source_image, source_label = source_image.cuda(), source_label.cuda()  # 32

            optimizer = optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            source_feature = encoder(source_image)

            # Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            class_loss.backward()
            optimizer.step()
            # if (batch_idx + 1) % C.epoch_verbose == 0:
            #     print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(batch_idx * len(source_image), len(source_train_loader.dataset), 100. * batch_idx / len(source_train_loader), class_loss.item()))
            total_loss += class_loss.item()
        
        total_losses.append(total_loss / len(source_train_loader))
        print('Loss: {:.6f}'.format(total_loss / len(source_train_loader)))
        Saccu, Taccu = test.tester(encoder, classifier, None, source_test_loader, target_test_loader, training_mode='source_only')
        print('Source Accuracy: {:.6f}\tTarget Accuracy: {:.6f}'.format(Saccu, Taccu))
        ACCU[0].append(Saccu)
        ACCU[1].append(Taccu)

        if (epoch + 1) % C.verbose == 0:
            # Saccu, Taccu = test.tester(encoder, classifier, None, source_test_loader, target_test_loader, training_mode='source_only')
            myplot({
                'title': 'Loss',
                'xlabel': 'Epoch',
                'ylabel': 'Loss',
                'data': {
                    'loss': [list(range(1,1+len(total_losses))), total_losses]
                },
                'savefig': 'saved_plot/source_loss.png'
            })
            myplot({
                'title': 'Accuracy',
                'xlabel': 'Epoch',
                'ylabel': 'Accuracy',
                'data': {
                    'Source': [list(range(1,1+len(ACCU[0]))), ACCU[0]],
                    'Target': [list(range(1,1+len(ACCU[1]))), ACCU[1]]
                },
                'savefig': 'saved_plot/source_accu.png'
            })
        if total_losses[-1] < bestloss:
            bestloss = total_losses[-1]
            bestaccu = Taccu
            print('Save on epoch {}, with accu = {:.6f}'.format(epoch, Taccu))
            save_model(encoder, classifier, None, 'source', save_name)
            # visualize(encoder, 'source', save_name, source_test_loader, target_test_loader)
    print('End of source only training, best accu = {:.6f}'.format(bestaccu))
    return bestaccu


def dann(encoder, classifier, discriminator, train_loaders, valid_loaders, save_name):
    source_train_loader, target_train_loader = train_loaders
    source_test_loader, target_test_loader = valid_loaders
    print("DANN training")
    
    classifier_criterion = nn.CrossEntropyLoss().cuda()
    discriminator_criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = optim.SGD(
    list(encoder.parameters()) +
    list(classifier.parameters()) +
    list(discriminator.parameters()),
    lr=C.lr,
    momentum=C.mom)
    
    total_losses = []
    class_losses = []
    domain_losses = []
    ACCU = [[], [], []]
    bestaccu = 0
    bestloss = 1e6
    for epoch in range(C.epochs):
        total_total_loss = 0
        total_class_loss = 0
        total_domain_loss = 0
        print('Epoch : {}'.format(epoch), end=' ')
        set_model_mode('train', [encoder, classifier, discriminator])

        start_steps = epoch * len(source_train_loader)
        total_steps = C.epochs * len(target_train_loader)
        
        for batch_idx, (source_data, target_data) in enumerate(zip(source_train_loader, target_train_loader)):

            source_image, source_label = source_data
            target_image, target_label = target_data

            p = float(batch_idx + start_steps) / total_steps
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # source_image = torch.cat((source_image, source_image, source_image), 1)

            source_image, source_label = source_image.cuda(), source_label.cuda()
            target_image, target_label = target_image.cuda(), target_label.cuda()
            combined_image = torch.cat((source_image, target_image), 0)

            optimizer = optimizer_scheduler(optimizer=optimizer, p=p)
            optimizer.zero_grad()

            combined_feature = encoder(combined_image)
            source_feature = encoder(source_image)

            # 1.Classification loss
            class_pred = classifier(source_feature)
            class_loss = classifier_criterion(class_pred, source_label)

            # 2. Domain loss
            domain_pred = discriminator(combined_feature, alpha)

            domain_source_labels = torch.zeros(source_label.shape[0]).type(torch.LongTensor)
            domain_target_labels = torch.ones(target_label.shape[0]).type(torch.LongTensor)
            domain_combined_label = torch.cat((domain_source_labels, domain_target_labels), 0).cuda()
            domain_loss = discriminator_criterion(domain_pred, domain_combined_label)

            total_loss = 2 * (class_loss * (1 - C.loss_coef) + domain_loss * C.loss_coef)
            total_loss.backward()
            optimizer.step()

            # if (batch_idx + 1) % C.epoch_verbose == 0:
            #     print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
            #         batch_idx * len(target_image), len(target_train_loader.dataset), 100. * batch_idx / len(target_train_loader), total_loss.item(), class_loss.item(), domain_loss.item()))
            total_class_loss += class_loss.cpu().item()
            total_domain_loss += domain_loss.cpu().item()
            total_total_loss += total_loss.cpu().item()

        total_losses.append((total_class_loss - total_domain_loss) / len(target_train_loader))
        class_losses.append(total_class_loss / len(target_train_loader))
        domain_losses.append(total_domain_loss / len(target_train_loader))
        print('Loss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format((total_class_loss - total_domain_loss) / len(target_train_loader), total_class_loss / len(target_train_loader), total_domain_loss / len(target_train_loader)))
        Saccu, Taccu, Daccu = test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='dann')
        print('Source Accuracy: {:.6f}\tTarget Accuracy: {:.6f}\tDomain Accuracy: {:.6f}'.format(Saccu, Taccu , Daccu))
        ACCU[0].append(Saccu)
        ACCU[1].append(Taccu)
        ACCU[2].append(Daccu)
        
        if (epoch + 1) % C.verbose == 0:
            # Saccu, Taccu, Daccu = test.tester(encoder, classifier, discriminator, source_test_loader, target_test_loader, training_mode='dann', verbose = True)
            myplot({
                'title': 'Loss',
                'xlabel': 'Epoch',
                'ylabel': 'Loss',
                'data': {
                    'total_loss': [list(range(1,1+len(total_losses))), total_losses],
                    'class_loss': [list(range(1,1+len(class_losses))), class_losses],
                    'domain_loss': [list(range(1,1+len(domain_losses))), domain_losses]
                },
                'savefig': 'saved_plot/dann_loss.png'
            })

            myplot({
                'title': 'Accuracy',
                'xlabel': 'Epoch',
                'ylabel': 'Accuracy',
                'data': {
                    'Source': [list(range(1,1+len(ACCU[0]))), ACCU[0]],
                    'Target': [list(range(1,1+len(ACCU[1]))), ACCU[1]],
                    'Domain': [list(range(1,1+len(ACCU[2]))), ACCU[2]]
                },
                'savefig': 'saved_plot/dann_accu.png'
            })
        if total_losses[-1] < bestloss:
            bestloss = total_losses[-1]
            bestaccu = Taccu
            print('Save on epoch {}, with accu = {:.6f}'.format(epoch, Taccu))
            save_model(encoder, classifier, discriminator, 'dann', save_name)
            # visualize(encoder, 'dann', save_name, source_test_loader, target_test_loader)
    print('End of DANN training, best accu = {:.6f}'.format(bestaccu))
    return bestaccu