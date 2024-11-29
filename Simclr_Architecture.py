from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = resnet50(pretrained=True).to('cuda')
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class HeadProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-1]) # remove the last layer
        self.fc1 = nn.Linear(1000, 200)
        self.fc2 = nn.Linear(200,100)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

Projection = HeadProjection().to('cuda')

class SIMCLR(nn.Module):
    def __init__(self,backbone,projection):
        super().__init__()
        self.backbone = backbone
        self.projection = Projection
    
    def forward(self,x):
        x=self.projection(self.backbone(x))
        return x

