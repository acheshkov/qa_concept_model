import torch
import math
import time
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


# CNN Model to detect 
class MyCNN2(torch.nn.Module):
    def __init__(self):
        super(MyCNN2, self).__init__()

        # paragraph feature extraction 
        self.cnn_paragraph = torch.nn.Sequential(
            torch.nn.Conv1d(1, 128, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(128, 128, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(16),

            torch.nn.Conv1d(128, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(8),

            torch.nn.Conv1d(64, 32, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
        )

        # span feature extraction 
        self.cnn_span = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(64, 64, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(10),

            torch.nn.Conv1d(64, 32, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(5),

            torch.nn.Conv1d(32, 16, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(16, 16, kernel_size=2, stride=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool1d(1)
        )

        self.fcn = torch.nn.Sequential(
            torch.nn.Linear(48, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, 1),
            torch.nn.Sigmoid()
        )
        # self.fc = torch.nn.Linear(16, 1)
        # self.sigmoid = torch.nn.Sigmoid()

        if torch.cuda.is_available(): self.cuda()
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.step = 0

    def forward(self, span, paragraph):
        out_span = self.cnn_span(span).view(-1, 16)
        out_paragraph = self.cnn_paragraph(paragraph).view(-1, 32)
        concat = torch.cat((out_paragraph, out_span), 1)
        return self.fcn(concat)

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
        for i, (X, Y, PARAGRAPH) in enumerate(dataloader):
            Y = Y.type(self.dtype).view(-1)
            W = ((Y == False).float() * pos_weigth) + ((Y == True).float()  * neg_weigth)
            X = Variable(X.type(self.dtype), requires_grad = False)
            Y = Variable(Y.float(), requires_grad = False)
            PARAGRAPH = Variable(PARAGRAPH.type(self.dtype), requires_grad = False)

            self.criterion.weight = W
            
            # Forward + Backward + Optimize
            self.optimizer.zero_grad()
            outputs = self.forward(X, PARAGRAPH).view(-1)
            loss = self.criterion(outputs, Y)
            loss.backward()
            self.optimizer.step()
            
            logger('loss', loss.data[0], self.step) # log to tensorboard
            self.step+=1
    
    def evaluateModel(self, dataloader):
        self.eval()
        
        positive = [(x,y,z) for (x,y,z) in dataloader if (y == True).sum() == 1]
        negative = [(x,y,z) for (x,y,z) in dataloader if (y == False).sum() == 1][0:len(positive)]
       
        correct = 0
        total = 0
        for X, Y, PARAGRAPH in positive + negative:
            Y = Y.view(-1).type(self.dtype)
            X = Variable(X.type(self.dtype), requires_grad = False)
            PARAGRAPH = Variable(PARAGRAPH.type(self.dtype), requires_grad = False)
            
            outputs = self.forward(X, PARAGRAPH).view(-1)
            correct += ((outputs.data > 0.5).float() == Y).sum()
            total += Y.size(0)
        
        print('Test Accuracy of the model on the %d test images: %d %%' % (total, 100 * correct / total))
        return correct / total