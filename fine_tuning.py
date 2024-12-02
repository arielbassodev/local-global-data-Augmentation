import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
import torch.nn.functional as F

class Module(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
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
