'''
Author: coffeecat
Date: 2025-04-05 23:04:28
LastEditors: Do not edit
LastEditTime: 2025-04-05 23:53:29
'''

import torch
import torch.nn as nn
vocab_size=100
bs,seq,hidden = 32,35,vocab_size

X = torch.ones(size=(bs,seq,hidden), dtype=torch.float32)

class RNN(nn.Module):
    def __init__(self, vocab_size,h_dim,o_dim):
        super().__init__()
        
        self.x_dim = vocab_size
        self.h_dim = h_dim
        self.o_dim = o_dim

        #h
        self.xh=nn.Linear(bias=True, in_features=vocab_size, out_features=h_dim)
        self.hh = nn.Linear(bias=True,in_features=h_dim,out_features=h_dim)
        #o
        self.ho = nn.Linear(bias=True,in_features=h_dim,out_features=o_dim)


    def forward(self,X, states=None):
        # X (bs,seq,vocab_size=hidden)
        bs,seq,hidden = X.shape
        inputs = X.transpose(0,1)
        # inputs (seq, bs, hidden)  seq可以理解为时间步

        if not states:
            states = torch.ones(size=(bs,self.h_dim),dtype=torch.float32)
        H = states # 当前时间步
        outputs = []        
        for x in inputs:
            # print(x.shape)
            # x (1,bs,hidden) torch.Size([32, 100])
            H = self.xh(x) + self.hh(H)
            O = self.ho(H)
            outputs.append(O)
        Y = torch.stack(outputs).transpose(0,1)
        return Y,H

print(X.shape)
# 创建 RNN 模型实例
model = RNN(vocab_size, h_dim=640, o_dim=10)
output, hidden_state = model(X)
print("Output shape:", output.shape)
print("Hidden state shape:", hidden_state.shape)


"""
torch.Size([32, 35, 100])
Output shape: torch.Size([32, 35, 10])
Hidden state shape: torch.Size([32, 640])
"""



