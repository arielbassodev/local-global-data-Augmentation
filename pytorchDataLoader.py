import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((255, 255))  
])

train_set = datasets.ImageFolder(root='C:/Users/abassoma/Documents/Dataset/Indonesian_dataset/indonesian_train_1',transform=transform)
val_set = datasets.ImageFolder(root='C:/Users/abassoma/Documents/Dataset/Indonesian_dataset/indonesian_train_2',transform=transform)
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_set,batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set,batch_size,shuffle=True)

