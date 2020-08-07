from torch import nn

class SDNet(nn.Module):
    def __init__(self, channels):
        super(SDNet, self).__init__()
        self.channels = channels

        # ConvLayers
        self.conv_1 = nn.Sequential(nn.Conv2d(self.channels, 96, 7, 2),
                                    nn.ReLU())
        # self.conv_10a = nn.Sequential(nn.Conv2d(512, 6, 1, 1),
        #                              nn.ReLU())
        self.conv_10 = nn.Sequential(nn.Conv2d(512, 6, 3, 1, padding=1),
                                      nn.ReLU())
        
        # PoolingLayers
        self.pool_1 = nn.MaxPool2d(3, 2)
        self.pool_4 = nn.MaxPool2d(3, 2)
        self.pool_8 = nn.MaxPool2d(3, 2)

        # DropoutLayers
        self.dropout = nn.Dropout2d(0.5)

        # FireLayers
        self.fire_2 = nn.Sequential(nn.Conv2d(96, 16, 1, 1),
                                    nn.ReLU())
        self.fire_2a = nn.Sequential(nn.Conv2d(16, 64, 1, 1),
                                     nn.ReLU())
        self.fire_2b = nn.Sequential(nn.Conv2d(16, 64, 3, 1, padding=1),
                                     nn.ReLU())
        self.fire_3 = nn.Sequential(nn.Conv2d(128, 16, 1, 1),
                                    nn.ReLU())
        self.fire_3a = nn.Sequential(nn.Conv2d(16, 64, 1, 1),
                                     nn.ReLU())
        self.fire_3b = nn.Sequential(nn.Conv2d(16, 64, 3, 1, padding=1),
                                     nn.ReLU())
        self.fire_4 = nn.Sequential(nn.Conv2d(128, 32, 1, 1),
                                    nn.ReLU())
        self.fire_4a = nn.Sequential(nn.Conv2d(32, 128, 1, 1),
                                     nn.ReLU())
        self.fire_4b = nn.Sequential(nn.Conv2d(32, 128, 3, 1, padding=1),
                                     nn.ReLU())
        self.fire_5 = nn.Sequential(nn.Conv2d(256, 32, 1, 1),
                                    nn.ReLU())
        self.fire_5a = nn.Sequential(nn.Conv2d(32, 128, 1, 1),
                                     nn.ReLU())
        self.fire_5b = nn.Sequential(nn.Conv2d(32, 128, 3, 1, padding=1),
                                     nn.ReLU())
        self.fire_6 = nn.Sequential(nn.Conv2d(256, 48, 1, 1),
                                    nn.ReLU())
        self.fire_6a = nn.Sequential(nn.Conv2d(48, 192, 1, 1),
                                     nn.ReLU())
        self.fire_6b = nn.Sequential(nn.Conv2d(48, 192, 3, 1, padding=1),
                                     nn.ReLU())
        self.fire_7 = nn.Sequential(nn.Conv2d(384, 48, 1, 1),
                                    nn.ReLU())
        self.fire_7a = nn.Sequential(nn.Conv2d(48, 192, 1, 1),
                                     nn.ReLU())
        self.fire_7b = nn.Sequential(nn.Conv2d(48, 192, 3, 1, padding=1),
                                     nn.ReLU())
        self.fire_8 = nn.Sequential(nn.Conv2d(384, 64, 1, 1),
                                    nn.ReLU())
        self.fire_8a = nn.Sequential(nn.Conv2d(64, 256, 1, 1),
                                     nn.ReLU())
        self.fire_8b = nn.Sequential(nn.Conv2d(64, 256, 3, 1, padding=1),
                                     nn.ReLU())
        self.fire_9 = nn.Sequential(nn.Conv2d(512, 64, 1, 1),
                                    nn.ReLU())
        self.fire_9a = nn.Sequential(nn.Conv2d(64, 256, 1, 1),
                                     nn.ReLU())
        self.fire_9b = nn.Sequential(nn.Conv2d(64, 256, 3, 1, padding=1),
                                     nn.ReLU())
    
    def concat(self, xa, xb):
        return torch.cat((xa, xb), dim=1)  

    def global_avg_pool(self, x):
        return x.mean([2,3])

    def feature_extract(self, x):
        # First ConvLayer
        x = self.conv_1(x)  # out: torch.Size([20, 96, 125, 125])
        x = self.pool_1(x)  # out: torch.Size([20, 96, 62, 62])

        # Fire 2
        x = self.fire_2(x)  # out: torch.Size([20, 16, 62, 62])
        xa = self.fire_2a(x)  # out: torch.Size([20, 64, 62, 62])
        xb = self.fire_2b(x)  # out: torch.Size([20, 64, 62, 62])
        x = self.concat(xa, xb)  # out: torch.Size([20, 128, 62, 62])

        # Fire 3
        x = self.fire_3(x)  # out: torch.Size([20, 16, 62, 62])
        xa = self.fire_3a(x)  # out: torch.Size([20, 64, 62, 62])
        xb = self.fire_3b(x)  # out: torch.Size([20, 64, 62, 62])
        x = self.concat(xa, xb)  # out: torch.Size([20, 128, 62, 62])

        # Fire 4
        x = self.fire_4(x)  # out: torch.Size([20, 32, 62, 62])
        xa = self.fire_4a(x)  # out: torch.Size([20, 128, 62, 62])
        xb = self.fire_4b(x)  # out: torch.Size([20, 128, 62, 62])
        x = self.concat(xa, xb)  # out: torch.Size([20, 256, 62, 62])
        x = self.pool_4(x)  # out: torch.Size([20, 256, 30, 30])

        # Fire 5
        x = self.fire_5(x)  # out: torch.Size([20, 32, 30, 30])
        xa = self.fire_5a(x)  # out: torch.Size([20, 128, 30, 30])
        xb = self.fire_5b(x)  # out: torch.Size([20, 128, 30, 30])
        x = self.concat(xa, xb)  # out: torch.Size([20, 256, 30, 30])

        # Fire 6
        x = self.fire_6(x)  # out: torch.Size([20, 48, 30, 30])
        xa = self.fire_6a(x)  # out: torch.Size([20, 192, 30, 30])
        xb = self.fire_6b(x)  # out: torch.Size([20, 192, 30, 30])
        x = self.concat(xa, xb)  # out: torch.Size([20, 384, 30, 30])

        # Fire 7
        x = self.fire_7(x)  # out: torch.Size([20, 48, 30, 30]) 
        xa = self.fire_7a(x)  # out: torch.Size([20, 192, 30, 30])
        xb = self.fire_7b(x)  # out: torch.Size([20, 192, 30, 30])
        x = self.concat(xa, xb)  # out: torch.Size([20, 384, 30, 30])

        # Fire 8
        x = self.fire_8(x)  # out: torch.Size([20, 64, 30, 30])
        xa = self.fire_8a(x)  # out: torch.Size([20, 256, 30, 30])
        xb = self.fire_8b(x)  # out: torch.Size([20, 256, 30, 30])
        x = self.concat(xa, xb)  # out: torch.Size([20, 512, 30, 30])
        x = self.pool_8(x)  # out: torch.Size([20, 512, 14, 14])

        # Fire 9
        x = self.fire_9(x)  # out: torch.Size([20, 64, 14, 14])
        xa = self.fire_9a(x)  # out: torch.Size([20, 256, 14, 14])
        xb = self.fire_9b(x)  # out: torch.Size([20, 256, 14, 14])
        x = self.concat(xa, xb)  # out: torch.Size([20, 512, 14, 14])

        return x

    def classifier(self, x):
        x = self.dropout(x)  # out: torch.Size([20, 512, 14, 14])
        x = self.conv_10(x)  # out: torch.Size([20, 6, 14, 14])
        x = self.global_avg_pool(x) # out: torch.Size([20, 6])

        return x

    def forward(self, x):
        fm = self.feature_extract(x)
        y = self.classifier(fm)        

        return y