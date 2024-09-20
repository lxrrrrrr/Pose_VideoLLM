import torch
from torch import nn
import numpy as np
import os
class motionnet(nn.Module):
    def __init__(self,hidden=2048,output=1408):
        super().__init__() 
        self.layer1 = nn.Linear(17*3,output)
        #self.layer2 = nn.Linear(hidden,output)

    def forward(self,x): 
        x=torch.flatten(x,1)
        x = self.layer1(x)
        #x = self.layer2(x)
        return x