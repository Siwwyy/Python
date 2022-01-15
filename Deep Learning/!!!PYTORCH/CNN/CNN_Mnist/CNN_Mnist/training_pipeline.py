import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(y_hat, y, verbose=False):

    assert y_hat.shape == y.shape, "Prediction shape: {} has to be equal to Target shape: {}".format(y_hat.shape, y.shape)
    # Test the model
    num_correct = torch.zeros(1, dtype=torch.float32, device=device)
    total = torch.tensor(y_hat.size(0), dtype=torch.float32, device=device)

    num_correct = torch.eq(y_hat, y).sum()
    total_accuracy = torch.div(num_correct, total) * 100.  # * 100%

    if verbose:
        print("Test Accuracy of the model: {:.2f}%".format(total_accuracy.item()))

    return total_accuracy

#Training
def train(model, train_loader, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        # gives batch data, normalize x when iterate train_loader
        images = images.to(device=device)   # batch x
        target = target.to(device=device)   # batch y

        # compute output
        output, conv2_out = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        model_pred_labels = torch.argmax(output, dim=1)
        acc = accuracy(model_pred_labels, target)
        losses.update(loss.item(), images.size(0))
        top1.update(acc.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % (images.size(0)) == 0:
            progress.display(i)

    return losses.avg


#Validation
def validate(model, val_loader, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            # gives batch data, normalize x when iterate train_loader
            images = images.to(device=device)   # batch x
            target = target.to(device=device)   # batch y

            # compute output
            output, conv2_out = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            model_pred_labels = torch.argmax(output, dim=1)
            acc = accuracy(model_pred_labels, target)
            losses.update(loss.item(), images.size(0))
            top1.update(acc.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % (images.size(0)) == 0:
                progress.display(i)

        ## TODO: this should also be done with the ProgressMeter
        #print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #      .format(top1=top1, top5=top5))

    return losses.avg



def main_pipeline(model:nn.Module, data_loader:dict, num_epochs:int=1, lr:float=0.001):
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)   
    cost_per_epoch = []
    valid_per_epoch = []
        
    total_step = len(data_loader['train'])

    overall_time = 0.0
    for epoch in range(num_epochs):
        
        #Train pass
        train_loss = train(model, data_loader['train'], criterion, optimizer, epoch + 1)
        cost_per_epoch.append(train_loss)

        valid_loss = validate(model, data_loader['valid'], criterion)
        valid_per_epoch.append(valid_loss)

    #print('Overall time: {:.1f} seconds'.format(overall_time))

    return cost_per_epoch, valid_per_epoch


def inference(model:nn.Module, data_loader:dict):
    # Test the model
    accumulated_accuracy = torch.zeros(1, dtype=torch.float32, device=device)
    iters = 0
    for i, (images, labels) in enumerate(data_loader['test']):
        # gives batch data, normalize x when iterate train_loader       
        b_x = images.to(device=device)   # batch x, inputs
        b_y = labels.to(device=device)   # batch y, targets

        output, conv2_out = model(b_x)

        # measure accuracy and record loss
        model_pred_labels = torch.argmax(output, dim=1)
        acc = accuracy(model_pred_labels, b_y)

        accumulated_accuracy += acc
        iters += 1

    total_accuracy = accumulated_accuracy / float(iters) #get a mean
    print("Test Accuracy of the model: {:.2f}%".format(total_accuracy.item()))

    return total_accuracy.item()