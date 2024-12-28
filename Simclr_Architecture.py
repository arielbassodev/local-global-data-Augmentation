from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = resnet50(pretrained=True).to('cpu')

from torchvision.models import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HeadProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 200)
        self.fc2 = nn.Linear(200,100)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

Projection = HeadProjection().to('cpu')

class SIMCLR(nn.Module):
    def __init__(self,projection, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection = Projection
    
    def forward(self,x):
        x=self.backbone(x)
        x=x.flatten(start_dim=1)
        x = self.projection(x)
        return x

backbone = resnet50(pretrained=True).to('cpu')
backbone = nn.Sequential(*list(backbone.children())[:-1])
simclr_model = SIMCLR(Projection,backbone)