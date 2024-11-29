from lightly.loss import NTXentLoss
from torchvision.models import resnet50
import torch
import torch.functional as F
model = resnet50(pretrained=True).to("cuda")

class MocoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = model
        self.fc1 = nn.Linear(1000,400)
        self.fc2 = nn.Linear(400, 200)
    def forward(self, x):
       x = self.encoder(x)
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       return x
encoder_query = MocoModel().to("cuda")
encoder_key = encoder_query
queue_dictionary = []

def momentum_step(m=1):
    params_q = encoder_query.state_dict()
    params_k = encoder_key.state_dict()
    
    dict_params_k = dict(params_k)
    
    for name in params_q:
        theta_k = dict_params_k[name]
        theta_q = params_q[name].data
        dict_params_k[name].data.copy_(m * theta_k + (1-m) * theta_q)