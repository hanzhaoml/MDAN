#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(ctx, inputs):
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


class MDANet(nn.Module):
    """
    Multi-layer perceptron with adversarial regularizer by domain classification.
    """
    def __init__(self, configs):
        super(MDANet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        self.num_domains = configs["num_domains"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i+1])
                                      for i in range(self.num_hidden_layers)]) for _ in range(self.num_domains)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        self.domains = nn.ModuleList([nn.Linear(self.num_neurons[-1], 2) for _ in range(self.num_domains)])
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer.apply for _ in range(self.num_domains)]

    def forward(self, sinputs, tinputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:
        """
        # instead of the hidden embedding layer extracting the features
        # we can have an adaptive model ensemble extracting the features
        # how to ensure continuous learning and add scalability to the mix
        # How to carry out multi source domain adaptation
        sh_relu, th_relu = sinputs, tinputs
        for i in range(self.num_domains):
            for hidden in self.hiddens[i]:
                sh_relu[i] = F.relu(hidden(sh_relu[i]))
            for hidden in self.hiddens[i]:
                th_relu[i] = F.relu(hidden(th_relu[i]))

        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_domains):
            logprobs.append(F.log_softmax(self.softmax(sh_relu[i]), dim=1))

        # Domain classification accuracies.
        sdomains, tdomains = [], []
        for i in range(self.num_domains):
            sdomains.append(F.log_softmax(self.domains[i](self.grls[i](sh_relu[i])), dim=1))
            tdomains.append(F.log_softmax(self.domains[i](self.grls[i](th_relu[i])), dim=1))
        return logprobs, sdomains, tdomains

    def inference(self, inputs):
        h_relu = inputs
        logprobs = []
        for i in range(self.num_domains):
            h_relu = inputs
            for hidden in self.hiddens[i]:
                h_relu = F.relu(hidden(h_relu))
            # Classification probability.
            softmax = self.softmax(h_relu)
            log_softmax = F.log_softmax(softmax, dim=1)
            logprobs.append(log_softmax)
        return logprobs