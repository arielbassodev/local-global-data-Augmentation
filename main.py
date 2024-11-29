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


ImageFile.LOAD_TRUNCATED_IMAGES = True
train_loader = loader.train_loader
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
