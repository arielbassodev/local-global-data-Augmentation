import matplotlib.pyplot as plt
from pytorch_metric_learning import losses
import torch.optim as optim
from PIL import ImageFile
import Simclr_Architecture as simclr
from lightly.loss import NTXentLoss
import Augmentation_method as augmentation
import pytorchDataLoader as loader
from tqdm import tqdm
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from lightly.data import LightlyDataset
from lightly.transforms import SimCLRTransform, utils


ImageFile.LOAD_TRUNCATED_IMAGES = True
train_loader = loader.train_loader
test_loader = loader.test_loader
simclr_model = simclr.SIMCLR(simclr.backbone,simclr.HeadProjection)
local_global = augmentation.local_global_augmentation
criterion = NTXentLoss()
optimizer = optim.SGD(simclr_model.parameters(), lr=0.001)
def training(train_loader,num_epochs, active_groups):
    epoch_losses = [] 
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        print(f"Début de l'époque {epoch + 1}")
        
        for step, (data, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            
            batch_1, batch_2 = local_global(data, active_groups)
            batch_1, batch_2 = batch_1.to('cuda'), batch_2.to('cuda')
            
            embedding_batch_1 = simclr_model(batch_1)
            embedding_batch_2 = simclr_model(batch_2)
            loss = criterion(embedding_batch_1,embedding_batch_2)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.grid(True)
    plt.legend()
    plt.savefig('Training_loss_simclr_others_essai_4.png')
    plt.show()
training(train_loader,70, active_groups=["ColorJitter"])



class classifier(nn.Module):
   def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(2048, num_classes)
   def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

for param in simclr_model.backbone.parameters():
    param.requires_grad = False
    
classifier_model = classifier(simclr_model.backbone, 31)

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
import torch.nn.functional as F

class Module(L.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = classifier_model
        # Définir la loss function
        self.criterion = nn.CrossEntropyLoss()
        num_classes = 31
        # Définir les métriques
        self.train_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.train_top3 = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        
        self.val_acc = TM.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_top3 = TM.Accuracy(task='multiclass', num_classes=num_classes, top_k=3)
        self.val_recall = TM.Recall(task='multiclass', num_classes=num_classes)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self.model(images)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self.model(images)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=-1)
        self.val_acc(preds, targets)
        self.val_recall(preds, targets)  
        self.log('val_loss', loss, on_step=True, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_recall', self.val_recall, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

my_module = Module()
trainer = L.Trainer(max_epochs=100)
trainer.fit(
    my_module,
    train_loader, 
    test_loader
)