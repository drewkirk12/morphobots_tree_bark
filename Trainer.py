import torch
import torch.nn as nn
import hyperparameters as hp

from model.model import Model
from torch.autograd import Variable
from dataset.data_loader import get_loader
from time import time
import os

class Trainer():
    def __init__(self):
        super().__init__()

    def create_model(self, model_name = hp.MODEL):
        self.criterion = nn.CrossEntropyLoss()

        self.net = Model(model_name, n_classes=hp.N_CLASSES)
        if (hp.PRETRAINED == True):
            self.net.load_state_dict(torch.load(hp.OLD_MODEL))

        self.freeze_layers(1)

        self.optimizer = torch.optim.Adam(self.net.parameters(), hp.LR)

    def freeze_layers(self, n):
        first_params = [self.net.model.conv1.parameters(), self.net.model.bn1.parameters()]
        layers = [self.net.model.layer1.parameters(), self.net.model.layer2.parameters(),
                  self.net.model.layer3.parameters(), self.net.model.layer4.parameters()]
        if n >= 1:
            for params in first_params:
                for param in params:
                    param.requires_grad = False

        for i in range(n - 1):
            layer = layers[i]
            for param in layer:
                param.requires_grad = False

    def get_loader(self, folder):
        loader, _ = get_loader(folder['train']['files'], folder['train']['labels'], hp.BATCH_SIZE)
        return loader

    def create_mini_batch(self, batch_loader):
        batch = next(batch_loader)[1]
        return Variable(batch[0]), Variable(batch[1].type(torch.LongTensor))

    def get_class_predictions(self, output, targets):
        return output.max(1)[1].type_as(targets)

    def update_iteration_info(self, batch_input, output, targets, epoch_acc, loader, loss, j, print_info=True):
        batch_size = batch_input.size(0)
        predictions = self.get_class_predictions(output, targets)
        correct = predictions.eq(targets)
        if not hasattr(correct, 'sum'):
            correct = correct.cpu()
        correct = correct.sum()
        acc = 100. * correct.item() / batch_size
        epoch_acc.append(acc)
        if print_info:
            print('\r', end='')
            print('{} / {} - {:.4f} - {:.2f}%'.format(j + 1, len(loader), loss.data[0], acc), end='',
                  flush=True)

        return epoch_acc
    
    def train(self, n_epoch, folder, model_name, print_info=True):
        loader = self.get_loader(folder)

        self.create_model(model_name)

        current_lr = hp.LR
        for i in range(n_epoch):
            start_time = time()
            data_loader = enumerate(loader)
            epoch_loss = []
            epoch_acc = []
            for j in range(len(loader)):
                batch_input, targets = self.create_mini_batch(data_loader)

                self.optimizer.zero_grad()
                try:
                    output = self.net(batch_input)
                except ValueError:
                    pass
                loss = self.criterion(output, targets)
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

                epoch_acc = self.update_iteration_info(batch_input, output, targets, epoch_acc, loader, loss, j,
                                                       print_info=print_info)

            print('\r', end='')
            end_time = time()
            self.update_epoch_info(i, n_epoch, current_lr, epoch_loss, epoch_acc, end_time - start_time, print_info=True)
            current_lr = self.update_lr(i, current_lr)

    def save_train_data(self):
        torch.save(self.net.state_dict(), hp.NEW_MODEL)

    def update_epoch_info(self, epoch, n_epoch, lr, loss, acc, epoch_time, print_info=True):
        if print_info:
            print('-----------------------------------------------------------')
            print('Epoch {} / {}'.format(epoch + 1, n_epoch))
            print('Lr: {}'.format(lr))
            print('Loss: {:.4f}'.format(sum(loss) / len(loss)))
            print('Accuracy: {:.2f}'.format(sum(acc) / len(acc)))
            print('Time: {:.2f} s'.format(epoch_time))
            print('-------------------------------------------------------------')

    def update_lr(self, epoch, old_lr):
        new_lr = old_lr
        if epoch in hp.EPOCH_LIST:
            for i, param_group in enumerate(self.optimizer.param_groups):
                new_lr = max(old_lr * hp.LR_DECAY, 0)
                param_group['lr'] = new_lr

        return new_lr
