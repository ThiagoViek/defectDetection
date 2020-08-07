import torch
from torch import nn
from torch.nn import functional as F 

class OSNet(nn.Module):
    def __init__(self, channels):
        """Constructor of the One-Shot Learning Siamese Network.
        channels : image channels (3 for RGB, 1 for Gray-Scale)
        """
        super(OSNet, self).__init__()
        self.channels = channels

        self.conv_net = nn.Sequential(
            nn.Conv2d(self.channels, 4, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 3),
            nn.ReLU(inplace=True), 
            nn.Conv2d(8, 8, 3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5) 
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * 194 * 194, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5)
        )

    def euclidean_dist(self, x_1, x_2):
        """Returns the Euclidean Distance (p=2) of the batch feature tensors
        from two network branches.
        x_1 : image 1
        x_2 : image 2
        """
        return F.pairwise_distance(x_1, x_2, p=2)
    
    def branch(self, x):
        """Returns the batch feature tensors from the given batch images.
        x : image
        """
        x = self.conv_net(x)
        x = x.view(-1, 8 * 194 * 194)
        x = self.fc(x)

        return x

    def forward(self, x_1, x_2):
        """Forward pass of the OSNet.
        x_1 : batch images to branch 1
        x_2 : batch images to branch 2
        """
        y_1 = self.branch(x_1)
        y_2 = self.branch(x_2)
        D = self.euclidean_dist(y_1, y_2)

        return D

class ContrastLoss(nn.Module):
    def __init__(self, margin=2):
        """Constructor of the Constructive Loss Function.
        margin : threshold value used to discriminate the Euclidean 
        Norm between two classes
        """
        super(ContrastLoss, self).__init__()
        self.margin = margin

    def forward(self, D, y):
        loss = 0.5 * y * D.pow(2) + (1 - y) * 0.5 * torch.clamp(self.margin - D, min=0).pow(2)
        return torch.mean(loss)