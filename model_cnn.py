import torch
import math
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


# CNN Model to detect 
class MyCNN(torch.nn.Module):
    def __init__(self, filters=0, kernel_size=3):
        super(MyCNN, self).__init__()
        # self.layer1 = torch.nn.Sequential(
        #     torch.nn.Conv1d(1, 64, kernel_size=kernel_size, stride=1),
        #     torch.nn.ReLU(),
        #     torch.nn.AdaptiveMaxPool1d(32),
        #     torch.nn.Conv1d(64, 32, kernel_size=kernel_size, stride=1),
        #     torch.nn.ReLU(),
        #     torch.nn.AdaptiveMaxPool1d(1)
        # )

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 32, kernel_size=kernel_size, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, kernel_size=kernel_size, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(32),
            torch.nn.Conv1d(32, 64, kernel_size=kernel_size, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
            #torch.nn.AdaptiveAvgPool1d(1)
        )
        self.fc = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

        if torch.cuda.is_available(): self.cuda()
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.step = 0

    def forward(self, X):
        out = self.layer1(X)
        #out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

    def trainModel(self, dataloader, pos_weigth, neg_weigth, logger,  learning_rate=0.001):
        """Train Model one epoch
        
        Arguments:
            dataloader {[type]} -- [description]
            pos_weigth {[type]} -- [description]
            neg_weigth {[type]} -- [description]
        
        Keyword Arguments:
            learning_rate {float} -- [description] (default: {0.001})
        """

        self.train()
        self.criterion = self.criterion if hasattr(self, "criterion") else torch.nn.BCELoss(size_average=False)
        self.optimizer = self.optimizer if hasattr(self, "optimizer") else torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        for i, (X, Y) in enumerate(dataloader):
            
            Y = Y.type(self.dtype).view(-1)
            W = ((Y == False).float() * pos_weigth) + ((Y == True).float()  * neg_weigth)
            X = Variable(X.type(self.dtype), requires_grad = False)
            Y = Variable(Y.float(), requires_grad = False)
            
            self.criterion.weight = W
            
            # Forward + Backward + Optimize
            self.optimizer.zero_grad()
            outputs = self.forward(X).view(-1)
            loss = self.criterion(outputs, Y)
            loss.backward()
            self.optimizer.step()
            
            logger('loss', loss.data[0], self.step) # log to tensorboard
            self.step+=1
    
    def evaluateModel(self, dataloader):
        self.eval()
        
        positive = [(x,y) for (x,y) in dataloader if (y == True).sum() == 1]
        negative = [(x,y) for (x,y) in dataloader if (y == False).sum() == 1][0:len(positive)]
       
        correct = 0
        total = 0
        for X, Y in positive + negative:
            Y = Y.view(-1).type(self.dtype)
            X = Variable(X.type(self.dtype), requires_grad = False)
            
            outputs = self.forward(X).view(-1)
            correct += ((outputs.data > 0.5).float() == Y).sum()
            total += Y.size(0)
        
        print('Test Accuracy of the model on the %d test images: %d %%' % (total, 100 * correct / total))
        return correct / total