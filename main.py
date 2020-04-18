'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np

import math
import sys
import time

import argparse
# from libs.layers import GaussianNoise
from data import FSIterator
from model import RNN
from train import train_main, test
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()

parser.add_argument('--logInterval', type=int, default=100, help='')
parser.add_argument('--saveModel', type=str, default="bestmodel", help='')
parser.add_argument('--savePath', type=str, default="png", help='')
parser.add_argument('--fileName', type=str, default="Adam0.01", help='')
parser.add_argument('--max_epochs', type=int, default=8, help='')
parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--hidden_size', type=int, default=8, help='')
parser.add_argument('--saveDir', type=str, default="png", help='')
parser.add_argument('--patience', type=int, default=5, help='')
parser.add_argument('--daytolook', type=int, default=15, help='')
parser.add_argument('--optim', type=str, default="Adam")  # Adam, SGD, RMSprop
parser.add_argument('--lr', type=float, metavar='LR', default=0.01,
                    help='learning rate (no default)')

args = parser.parse_args()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == "__main__":
    # prepare data
    batch_size = args.batch_size
    n_epoches = args.max_epochs

    device = torch.device("cuda")

    # setup model

    input_size = 10
    hidden_size = args.hidden_size
    output_size = 1
    print("Set model") 
    model = RNN(input_size, hidden_size, output_size, batch_size).to(device)

    # define loss
    print("Set loss and Optimizer")
    criterion = nn.MSELoss(reduction='none')

    # define optimizer
    optimizer = "optim." + args.optim
    optimizer = eval(optimizer)(model.parameters(), lr=args.lr)

    logInterval = args.logInterval
    current_loss = 0
    all_losses = []

    start = time.time()

    patience = args.patience
    savePath = args.savePath

    # train_path = "../data/dummy/classification_test.csv"
    train_path = "../data/regression/train"
    test_path = "../data/regression/test"
    valid_path = "../data/regression/valid"

    print("Train starts")
    for ei in range(args.max_epochs):
        bad_counter = 0
        best_loss = -1.0

        train_main(args, model, train_path, criterion, optimizer)
        dayloss, valid_loss, dayerr, valid_err = test(args, model, valid_path, criterion)

        print("valid loss : {}".format(valid_loss))
        print(dayloss)

        if valid_loss < best_loss or best_loss < 0:
            print("find best")
            best_loss = valid_loss
            bad_counter = 0
            torch.save(model, args.saveModel)
        else:
            bad_counter += 1

        if bad_counter > patience:
            print('Early Stopping')
            break

    print("------------test-----------------")
    dayloss, valid_loss, dayerr, valid_err  = test(args, model, test_path, criterion)
    print("valid loss : {}".format(valid_loss))
    print(dayloss)
    
    print("valid error: {}".format(valid_err))
    print(dayerr)
    
    # write to csv
    import csv
    f = open('Linear.csv', 'w', encoding='utf-8', newline='')
    #f = open('Gaussian.csv', 'w', encoding='utf-8', newline='')
    #f = open('Normalization.csv', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(["batch_size: " + str(args.batch_size), " max_epoches: " + str(args.max_epochs), " hidden_size: " + str(args.hidden_size),
                 " patience: " + str(args.patience), " daytolook: " + str(args.daytolook), " optimizer: "+args.optim, " learning_rate: " + str(args.lr)])
    wr.writerow(["dayloss", dayloss.tolist()])
    wr.writerow(["valid_loss", valid_loss])

    wr.writerow(["dayerror", dayerr.tolist()])
    wr.writerow(["valid_error", valid_err])
    f.close()
    
    '''
    result_matrix = pd.DataFrame(
        data={"dayloss": list(dayloss.data), "valid_loss": valid_loss})
    result_matrix.to_csv("test.csv", index=False, header=True)
    '''
    
    # draw a plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dataset = [dayloss.tolist(), dayerr.tolist()]
    category = list()
    daylength = []
    for i in range(args.daytolook):
        daylength.append(' ')
    
    data = {'day': daylength,  'Linear_MSE_loss' : dataset[0], 'Linear_Error_rate': dataset[1]}
    #data = {'day': daylength,  'Gaussian_MSE_loss' : dataset[0], 'Gaussian_Error_rate': dataset[1]}
    #data = {'day': daylength,  'Normal_MSE_loss' : dataset[0], 'Normal_Error_rate': dataset[1]}

    df1 = pd.DataFrame(data=data)
    df2 = pd.DataFrame(data=data)

    ax = plt.gca()

    # draw and save file
    df1.plot(kind='line', x='day', y = 'Linear_MSE_loss', ax=ax, color='blue')
    #df1.plot(kind='line', x='day', y = 'Gaussian_MSE_loss', ax=ax, color='blue')
    #df1.plot(kind='line', x='day', y = 'Normal_MSE_loss', ax=ax, color='blue')
    save_path = os.path.join("./", args.saveDir)
    ax.figure.savefig(save_path + "/linear_MSE_loss.png")
    #ax.figure.savefig(save_path + "/gaussian_MSE_loss.png")
    #ax.figure.savefig(save_path + "/normal_MSE_loss.png")

    plt.close()

    ax = plt.gca()

    df2.plot(kind='line', x='day', y = 'Linear_Error_rate', ax=ax, color='red')
    #df2.plot(kind='line', x='day', y = 'Gaussian_Error_rate', ax=ax, color='red')
    #df2.plot(kind='line', x='day', y = 'Normal_Error_rate', ax=ax, color='red')
    save_path = os.path.join("./", args.saveDir)
    ax.figure.savefig(save_path + "/linear_err_rate.png")
    #ax.figure.savefig(save_path + "/gaussian_err_rate.png")
    #ax.figure.savefig(save_path + "/normal_err_rate.png")

    plt.close()
    