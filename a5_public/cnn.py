#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, embed_size, m_word, k, f):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(embed_size, f, k)
        self.maxpool = nn.MaxPool1d(kernel_size=m_word - k + 1)

    def forward(self, x_reshaped: torch.tensor) -> torch.tensor:
        x_conv_out = self.conv(x_reshaped)
        x_conv_out = self.maxpool(F.relu(x_conv_out))

        return torch.squeeze(x_conv_out, -1)


### END YOUR CODE
