import torch as torch 
import torch.nn as nn


import torch 
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, input_size:int, action_size:int, hidden_size:int=256,non_linear:nn.Module=nn.ReLU):
        """
        input: int
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        hidden_size: int
            The number of neurons in the hidden layer

        This is a seperate class because it may be useful for the bonus questions
        """
        super(MLP, self).__init__()
        #====== TODO: ======
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_size, hidden_size) #wouldn't I need a Flatten() before this?
        self.output = nn.Linear(hidden_size, action_size) #output layer
        self.non_linear = non_linear()

        # self.model = nn.Sequential(OrderedDict([
        #     ('flatten', nn.Flatten()),
        #     ('linear1', nn.Linear(input_size, hidden_size)),
        #     ('non_linear', non_linear()),
        #     ('output', nn.Linear(hidden_size, action_size))
        # ]))

    def forward(self, x:torch.Tensor)->torch.Tensor:
        #====== TODO: ======
        x = self.flatten(x)
        x = self.non_linear(self.linear1(x))
        x = self.output(x)
        return x
        # return self.model(x)

class Nature_Paper_Conv(nn.Module):
    """
    A class that defines a neural network with the following architecture:
    - 1 convolutional layer with 32 8x8 kernels with a stride of 4x4 w/ ReLU activation
    - 1 convolutional layer with 64 4x4 kernels with a stride of 2x2 w/ ReLU activation
    - 1 convolutional layer with 64 3x3 kernels with a stride of 1x1 w/ ReLU activation
    - 1 fully connected layer with 512 neurons and ReLU activation. 
    - Same as 'Nature_Paper' but with dropout with keep probability .5 on 2nd convolutional layer
    Based on 2015 paper 'Human-level control through deep reinforcement learning' by Mnih et al
    """
    def __init__(self, input_size:tuple[int], action_size:int,**kwargs):
        """
        input: tuple[int]
            The input size of the image, of shape (channels, height, width)
        action_size: int
            The number of possible actions
        **kwargs: dict
            additional kwargs to pass for stuff like dropout, etc if you would want to implement it
        """
        super(Nature_Paper_Conv, self).__init__()
        #===== TODO: ======
        # self.CNN = nn.Sequential(*[
        #     nn.Conv2d(input_size[0], 32, (8,8), stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, (4,4), stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, (3,3), stride=1),
        #     nn.ReLU()
        # ])
        # print(input_size)
        self.CNN = nn.Sequential(OrderedDict([
          ('0', nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4)),
          ('relu0', nn.ReLU()),
          ('2', nn.Conv2d(32, 64, kernel_size=4, stride=2)),
          # ('dropout', nn.Dropout(p=0.5)),
          ('relu1', nn.ReLU()),
          ('4', nn.Conv2d(64, 64, kernel_size=3, stride=1)),
          ('relu2', nn.ReLU())
        ]))

        conv_output_size = 64 * 7 * 7

        # self.MLP = nn.Sequential(
        #   nn.Linear(conv_output_size,512),
        #   nn.ReLU(),
        #   nn.Linear(512, action_size)
        # ) #MLP((64,3,3), action_size, 512)

        self.MLP = MLP(conv_output_size, action_size, hidden_size=512)



    def forward(self, x:torch.Tensor)->torch.Tensor:
        #==== TODO: ====
        return self.MLP(self.CNN(x))

        
        
    