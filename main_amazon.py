#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time
import argparse
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from model import MDANet
from utils import get_logger
from utils import data_loader
from utils import multi_data_loader


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="amazon")
parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                    type=float, default=1.0)
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-v", "--verbose", help="Verbose mode: True -- show training progress. False -- "
                                            "not show training progress.", type=bool, default=True)
parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                    type=str, default="mdan")
# The experimental setting of using 5000 dimensions of features is according to the papers in the literature.
parser.add_argument("-d", "--dimension", help="Number of features to be used in the experiment",
                    type=int, default=5000)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                    type=float, default=1e-2)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=15)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=20)
parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str, default="dynamic")
# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(args.name)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Loading the randomly partition the amazon data set.
time_start = time.time()
amazon = np.load("./amazon.npz")
amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                       shape=amazon['xx_shape'][::-1]).tocsc()
amazon_xx = amazon_xx[:, :args.dimension]
amazon_yy = amazon['yy']
amazon_yy = (amazon_yy + 1) / 2
amazon_offset = amazon['offset'].flatten()
time_end = time.time()
logger.info("Time used to process the Amazon data set = {} seconds.".format(time_end - time_start))
logger.info("Number of training instances = {}, number of features = {}."
             .format(amazon_xx.shape[0], amazon_xx.shape[1]))
logger.info("Number of nonzero elements = {}".format(amazon_xx.nnz))
logger.info("amazon_xx shape = {}.".format(amazon_xx.shape))
logger.info("amazon_yy shape = {}.".format(amazon_yy.shape))
# Partition the data into four categories and for each category partition the data set into training and test set.
data_name = ["books", "dvd", "electronics", "kitchen"]
num_data_sets = 4
data_insts, data_labels, num_insts = [], [], []
for i in range(num_data_sets):
    data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i+1], :])
    data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
    logger.info("Length of the {} data set label list = {}, label values = {}, label balance = {}".format(
        data_name[i],
        amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :].shape[0],
        np.unique(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :]),
        np.sum(amazon_yy[amazon_offset[i]: amazon_offset[i+1], :])
    ))
    num_insts.append(amazon_offset[i+1] - amazon_offset[i])
    # Randomly shuffle.
    r_order = np.arange(num_insts[i])
    np.random.shuffle(r_order)
    data_insts[i] = data_insts[i][r_order, :]
    data_labels[i] = data_labels[i][r_order, :]
logger.info("Data sets: {}".format(data_name))
logger.info("Number of total instances in the data sets: {}".format(num_insts))
# Partition the data set into training and test parts, following the convention in the ICML-2012 paper, use a fixed
# amount of instances as training and the rest as test.
num_trains = int(2000 * args.frac)
input_dim = amazon_xx.shape[1]
# The confusion matrix stores the prediction accuracy between the source and the target tasks. The row index the source
# task and the column index the target task.
results = {}
logger.info("Training fraction = {}, number of actual training data instances = {}".format(args.frac, num_trains))
logger.info("-" * 100)

if args.model == "mdan":
    configs = {"input_dim": input_dim, "hidden_layers": [1000, 500, 100], "num_classes": 2,
               "num_epochs": args.epoch, "batch_size": args.batch_size, "lr": 1.0, "mu": args.mu, "num_domains":
                   num_data_sets - 1, "mode": args.mode, "gamma": 10.0, "verbose": args.verbose}
    num_epochs = configs["num_epochs"]
    batch_size = configs["batch_size"]
    num_domains = configs["num_domains"]
    lr = configs["lr"]
    mu = configs["mu"]
    gamma = configs["gamma"]
    mode = configs["mode"]
    logger.info("Training with domain adaptation using PyTorch madnNet: ")
    logger.info("Hyperparameter setting = {}.".format(configs))
    error_dicts = {}
    for i in range(num_data_sets):
        # Build source instances.
        source_insts = []
        source_labels = []
        for j in range(num_data_sets):
            if j != i:
                source_insts.append(data_insts[j][:num_trains, :].todense().astype(np.float32))
                source_labels.append(data_labels[j][:num_trains, :].ravel().astype(np.int64))
        # Build target instances.
        target_idx = i
        target_insts = data_insts[i][num_trains:, :].todense().astype(np.float32)
        target_labels = data_labels[i][num_trains:, :].ravel().astype(np.int64)
        # Train DannNet.
        mdan = MDANet(configs).to(device)
        optimizer = optim.Adadelta(mdan.parameters(), lr=lr)
        mdan.train()
        # Training phase.
        time_start = time.time()
        for t in range(num_epochs):
            running_loss = 0.0
            train_loader = multi_data_loader(source_insts, source_labels, batch_size)
            for xs, ys in train_loader:
                slabels = torch.ones(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                tlabels = torch.zeros(batch_size, requires_grad=False).type(torch.LongTensor).to(device)
                for j in range(num_domains):
                    xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                    ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)
                ridx = np.random.choice(target_insts.shape[0], batch_size)
                tinputs = target_insts[ridx, :]
                tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
                optimizer.zero_grad()
                logprobs, sdomains, tdomains = mdan(xs, tinputs)
                # Compute prediction accuracy on multiple training sources.
                losses = torch.stack([F.nll_loss(logprobs[j], ys[j]) for j in range(num_domains)])
                domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                           F.nll_loss(tdomains[j], tlabels) for j in range(num_domains)])
                # Different final loss function depending on different training modes.
                if mode == "maxmin":
                    loss = torch.max(losses) + mu * torch.min(domain_losses)
                elif mode == "dynamic":
                    loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
                else:
                    raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            logger.info("Iteration {}, loss = {}".format(t, running_loss))
        time_end = time.time()
        # Test on other domains.
        mdan.eval()
        target_insts = torch.tensor(target_insts, requires_grad=False).to(device)
        target_labels = torch.tensor(target_labels)
        preds_labels = torch.max(mdan.inference(target_insts), 1)[1].cpu().data.squeeze_()
        pred_acc = torch.sum(preds_labels == target_labels).item() / float(target_insts.size(0))
        error_dicts[data_name[i]] = preds_labels.numpy() != target_labels.numpy()
        logger.info("Prediction accuracy on {} = {}, time used = {} seconds.".
                    format(data_name[i], pred_acc, time_end - time_start))
        results[data_name[i]] = pred_acc
    logger.info("Prediction accuracy with multiple source domain adaptation using madnNet: ")
    logger.info(results)
    pickle.dump(error_dicts, open("{}-{}-{}-{}.pkl".format(args.name, args.frac, args.model, args.mode), "wb"))
    logger.info("*" * 100)
else:
    raise ValueError("No support for the following model: {}.".format(args.model))

