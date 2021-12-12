import torch
from torch import nn
from torch.utils.data import DataLoader, dataloader, dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import torchvision.models as models

'''
FashionMNIST_info:
 Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples. 
 Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.
 input_dimension: 28*28
 output_dimension(type): 10
'''

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

#The data structure of FashionMNIST organised as Tensor plus label
#Method dataloader in torch helps to convert the data in a more convinent way to use

batch_size = 64
train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

#Construct the Structure of MLP
#Containing 2 Fully Connected Layer,using ReLU() as activation function.
device = "cuda" if torch.cuda.is_avaliable() else "cpu"
print(f"Using {device} device")
class BPNet(nn.Module):
    def __init__(self):
        super(BPNet,self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = BPNet().to(device)
print(model)

#Define loss and training process
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
        optimizer.step() #Using Stochastic Gradient Descent Method

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
for t in range(0,3):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

torch.save(model,'mymodel.pth')

'''
Run on GTX 2080
Result：
Epoch 50
-------------------------------
0.356011 [    0/60000]
0.487588 [ 6400/60000]
0.313108 [12800/60000]
0.533249 [19200/60000]
0.472418 [25600/60000]
0.483078 [32000/60000]
0.484275 [38400/60000]
0.676488 [44800/60000]
0.594812 [51200/60000]
0.431610 [57600/60000]
Test Error: 
 Accuracy: 82.7%, Avg loss: 0.488310 

'''
