""" 
--------------------- RESNET FILE ---------------------
Author: Reza Tanakizadeh
Year  : 2022
P_name: Sima face verification project
Desc  : This file a an Inplemented of resnet network (Resnet18, ...)
-------------------------------------------------------
"""
import torch.nn as nn

# --- Define Residual block
class ResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, downsample=1):
    super().__init__()
    # --- Variables
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.downsample = downsample

    # --- Residual parts
    # --- Conv part
    self.blocks = nn.Sequential(OrderedDict(
        {
            # --- First Conv
            'conv1' : nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.downsample, padding=1, bias=False),
            'bn1'   : nn.BatchNorm2d(self.out_channels),
            'Relu1' : nn.ReLU(),
         
            # --- Secound Conv
            'conv2' : nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            'bn2'   : nn.BatchNorm2d(self.out_channels)
        }
    ))
    # --- shortcut part
    self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=self.downsample, bias=False),
            'bn'   : nn.BatchNorm2d(self.out_channels)
        }
    ))
  
  # --- Forward Inplementation
  def forward(self, x):
    residual = x
    if (self.in_channels != self.out_channels) : residual = self.shortcut(x)
    x = self.blocks(x)
    x += residual
    return x


# --- Make ResNet18
class ResNet18(nn.Module):
  def __init__(self):
    super().__init__()

    # --- Pre layers with 7*7 conv with stride2 and a max-pooling
    self.PreBlocks = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # --- Define all Residual Blocks here
    self.CoreBlocka = nn.Sequential(
        ResidualBlock(64,64 ,downsample=1),
        ResidualBlock(64,64 ,downsample=1),

        ResidualBlock(64,128 ,downsample=2),
        ResidualBlock(128,128 ,downsample=1),

        ResidualBlock(128,256 ,downsample=2),
        ResidualBlock(256,256 ,downsample=1),

        ResidualBlock(256,512 ,downsample=2),
        ResidualBlock(512,512 ,downsample=1)
    )

    # --- Make Average pooling
    self.avg = nn.AdaptiveAvgPool2d((1,1))

    # --- FC layer for output
    self.fc = nn.Linear(512, 512, bias=False)

  def forward(self, x):
    x = self.PreBlocks(x)
    x = self.CoreBlocka(x)
    x = self.avg(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = F.normalize(x, p=2, dim=1)
    return x



# --- Make ResNet18
class ResNet34(nn.Module):
  def __init__(self):
    super().__init__()

    # --- Pre layers with 7*7 conv with stride2 and a max-pooling
    self.PreBlocks = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    # --- Define all Residual Blocks here
    self.CoreBlocka = nn.Sequential(
        ResidualBlock(64,64 ,downsample=1),
        ResidualBlock(64,64 ,downsample=1),
        ResidualBlock(64,64 ,downsample=1),

        ResidualBlock(64,128 ,downsample=2),
        ResidualBlock(128,128 ,downsample=1),
        ResidualBlock(128,128 ,downsample=1),
        ResidualBlock(128,128 ,downsample=1),

        ResidualBlock(128,256 ,downsample=2),
        ResidualBlock(256,256 ,downsample=1),
        ResidualBlock(256,256 ,downsample=1),
        ResidualBlock(256,256 ,downsample=1),
        ResidualBlock(256,256 ,downsample=1),
        ResidualBlock(256,256 ,downsample=1),

        ResidualBlock(256,512 ,downsample=2),
        ResidualBlock(512,512 ,downsample=1),
        ResidualBlock(512,512 ,downsample=1)
    )

    # --- Make Average pooling
    self.avg = nn.AdaptiveAvgPool2d((1,1))

    # --- FC layer for output
    self.fc = nn.Linear(512, 512, bias=False)

  def forward(self, x):
    x = self.PreBlocks(x)
    x = self.CoreBlocka(x)
    x = self.avg(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    x = F.normalize(x, p=2, dim=1)
    return x