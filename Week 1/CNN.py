import torch
from torch import nn
from torch.nn.modules import padding
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.conv import Conv2d
from torch.utils.data import DataLoader, dataloader, dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.models as models

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)


batch_size = 64
train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

#The Structure of CNN
# 2 convoluntional layers using ReLU() as activation function and MaxPooll() as pooling 
# and one FC layer
device = "cuda " if torch.cuda.is_avaliable else "cpu"
class GNN_model(nn.Module):
    def __init__(self):
        super(GNN_model,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=4,kernel_size=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels=4,out_channels=8,kernel_size=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten(),
            nn.Linear(8*7*7,10),
        )
    def forward(self,x):
        logits = self.linear_relu_stack(x)
        return logits

model = GNN_model().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
def train(dataloader,model,loss_fn,optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch,(x,y) in enumerate(dataloader):
        x,y = x.to(device),y.to(device)
        pred = model(x)
        loss = loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        if batch %100==0:
            loss,current = loss.item(),batch*len(x)
            print(f"{loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader,model,loss_fn):
    size = len(dataloader.dataset)
    num_batches =len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            pred = model(x)
            test_loss += loss_fn(pred,y).item()
            correct +=(pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /=num_batches
    correct /=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#--main program--
for t in range(0,10):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

torch.save(model,'mymodel.pth')

'''
Result：

Epoch 50
-------------------------------
0.345739 [    0/60000]
0.410217 [ 6400/60000]
0.275213 [12800/60000]
0.511766 [19200/60000]
0.419549 [25600/60000]
0.518529 [32000/60000]
0.452981 [38400/60000]
0.594299 [44800/60000]
0.598589 [51200/60000]
0.438517 [57600/60000]
Test Error: 
 Accuracy: 83.7%, Avg loss: 0.456537

'''
