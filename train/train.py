import time

import torch
import torch.nn as nn
import torch.optim as optim

from data import FSIterator

import numpy as np
import pandas as pd
import sys


def train_main(args, model, train_path, criterion, optimizer):
    iloop = 0
    current_loss = 0
    all_losses = []
    batch_size = args.batch_size
    train_iter = FSIterator(train_path, batch_size)

    for input, target, mask in train_iter:  # TODO for debugging
        loss = train(args, model, input, mask, target, optimizer, criterion)
        current_loss += loss

        if (iloop+1) % args.logInterval == 0:
            print("%d %.4f" % (iloop+1, current_loss/args.logInterval))
            all_losses.append(current_loss / args.logInterval)
            current_loss = 0

        iloop += 1

    


def train(args, model, input, mask, target, optimizer, criterion):
    model = model.train()
    loss_matrix = []
    optimizer.zero_grad()

    output = model(input)

    for t in range(input.size(0)-1):
        loss = criterion(output[t].view(args.batch_size, -1),
                         input[t+1][:, 0].view(args.batch_size, -1))
        loss_matrix.append(loss.view(1, -1))

    loss_matrix = torch.cat(loss_matrix, dim=0)

    masked = loss_matrix * mask[1:]

    loss = torch.sum(masked) / torch.sum(mask[1:])

    loss.backward()

    optimizer.step()

    return loss.item()


def evaluate(args, model, input, mask, target, criterion):
    
    loss_matrix = []
    err_matrix = []
    daylen = args.daytolook
    output = model(input)

    '''Part of loss'''
    for t in range(input.size(0)-1):
        loss = criterion(output[t].view(args.batch_size, -1),
                         input[t+1][:, 0].view(args.batch_size, -1))
        err = abs((input[t+1][:, 0].view(args.batch_size, -1) - output[t].view(args.batch_size, -1))/(input[t+1][:, 0].view(args.batch_size, -1)+sys.float_info.epsilon))

        loss_matrix.append(loss.view(1, -1))
        err_matrix.append(err.view(1,-1))

    loss_matrix = torch.cat(loss_matrix, dim=0)
    err_matrix = torch.cat(err_matrix, dim=0)
    
    masked_loss = loss_matrix * mask[1:]  # TODO
    masked_err = err_matrix * mask[1:]

    '''
    masked_err[masked_err == float('inf')] = np.nan
    masked_err[np.isnan(masked_err.cpu())] =np.nanmax(masked_err.cpu())
    '''

    lossPerDay = torch.sum(masked_loss, dim=1) / \
        torch.sum(mask[1:], dim=1)  # 1*daylen
    errPerDay = torch.sum(masked_err, dim=1) / \
        torch.sum(mask[1:], dim=1)  # 1*daylen
    
    loss = torch.sum(masked_loss[:daylen]) / torch.sum(mask[1:][:daylen])
    err = torch.sum(masked_err[:daylen]) / torch.sum(mask[1:][:daylen])

    
    return lossPerDay, loss.item(), errPerDay, err.item()


def test(args, model, test_path, criterion):
    current_loss = 0
    current_err = 0
    lossPerDays = []
    errPerDays = []
    lossPerDays_avg = []
    errPerDays_avg = []

    model.eval()

    daylen = args.daytolook
    with torch.no_grad():
        iloop = 0
        test_iter = FSIterator(test_path, args.batch_size, 1)
        for input, target, mask in test_iter:

            lossPerDay, loss, errPerDay, err = evaluate(
                args, model, input, mask, target, criterion)
            lossPerDays.append(lossPerDay[:daylen])  # n_batches * 10
            errPerDays.append(errPerDay[:daylen])
            current_loss += loss
            current_err += err
            iloop += 1

        lossPerDays = torch.stack(lossPerDays)
        errPerDays = torch.stack(errPerDays)
        lossPerDays_avg = lossPerDays.sum(dim=0)
        errPerDays_avg = errPerDays.sum(dim=0)
        lossPerDays_avg = lossPerDays_avg/iloop
        errPerDays_avg = errPerDays_avg/iloop

        current_loss = current_loss/iloop
        current_err = current_err/iloop
            #TODO errPerDays_avg, current_loss
    return lossPerDays_avg, current_loss, errPerDays_avg, current_err
