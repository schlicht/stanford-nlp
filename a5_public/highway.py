#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, embed_size):
        super(Highway, self).__init__()
        self.W_proj = nn.Linear(embed_size, embed_size, bias = True)
        self.W_gate = nn.Linear(embed_size, embed_size, bias = True)


    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        x_proj = F.relu(self.W_proj(x_conv_out))
        x_gate = torch.sigmoid(self.W_gate(x_conv_out))
        x_highway = torch.mul(x_proj, x_gate) + torch.mul(x_conv_out, 1 - x_gate)
        return x_highway
### END YOUR CODE
