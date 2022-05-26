#This file use to recomplement the classic model of Computer vision MoCo
#And due to the traditional data ImageNet is too large, in this small project
#we are going to use the dataset FashionMNIST provided by pytorch to test it.
import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, dataloader, dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import os
#os.environ['CUDA_VISIBLE_DIVICES']= '2'
batch_size = 60
device = "cuda"
epoch = 500
class MLP(torch.nn.Module):
    def __init__(self,input_dim,class_num):
        super(MLP,self).__init__()
        self.dim = input_dim
        self.num = class_num
    def forward(self,x):
        x = torch.nn.Linear(self.dim,self.num)(x)
        return x
class encoder(torch.nn.Module):
    def __init__(self,input_dim):
        super(encoder,self).__init__()
        self.pooling = torch.nn.MaxPool2d(kernel_size=2,stride=2)
        self.layer1 = torch.nn.Conv2d(1,1,kernel_size=4,padding=2)
        self.layer2 = torch.nn.Conv2d(1,1,kernel_size=4,padding=2)
        self.activation = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
    def forward(self,x):
        x = self.pooling(self.activation(self.layer1(x)))
        x = self.pooling(self.activation(self.layer2(x)))
        return self.flatten(x)

class MOCO(torch.nn.Module):
    def __init__(self,input_dim,dict_size=3000,momentum=0.99,T=0.07):
        super(MOCO, self).__init__()
        self.K = dict_size
        self.M = momentum
        self.dim = input_dim
        self.T = T
        self.outdim = 49#magic number 4 is determined by 2 layer of pooling in class encoder.
        self.encoder_q = encoder(input_dim)
        self.encoder_k = encoder(input_dim)
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.register_buffer("queue", torch.randn(self.outdim, self.K))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_encoder_k(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.M + param_q.data * (1. - self.M)

    @torch.no_grad()
    def deQ_enQ(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = torch.nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self.update_encoder_k()  # update the key encoder
            k = self.encoder_k(im_k)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators,positives are the 0-th
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self.deQ_enQ(k)

        return logits, labels
def image_augment(image):
    x = torch.einsum('...ij->...ji',[image])#equal to rotate the image.
    return x

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
train_dataloader = DataLoader(training_data,batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

model = MOCO(input_dim=28).to(device)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
classifier = MLP(input_dim=49,class_num=10).to(device)

def train(dataloader,model,optimizer):
    criteria = torch.nn.CrossEntropyLoss()
    size = len(dataloader.dataset)
    for batch,(x,label) in enumerate(dataloader):
        x = x.to(device)
        x_aug = image_augment(x)
        logits,labels = model(x,x_aug)
        loss = criteria(logits,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch %100==0:
            loss,current = loss.item(),batch*len(x)
            print(f"{loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader,model,classifier):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss,correct = 0,0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            pred = classifier(model.encoder_q(x))
            test_loss += torch.nn.CrossEntropyLoss()(pred,y).item()
            correct +=(pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /=num_batches
    correct /=size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n") 

#Pre_train
for x in range(1,20):
    print(x)
    train(train_dataloader,model,optimizer)

#Train for classifier
for x in range(1,20):
    size = len(dataloader.dataset)
    classifier.train()
    for batch,(x,y) in enumerate(dataloader):
        x,y = x.to(device),y.to(device)
        pred = model(x)
        loss = torch.nn.CrossEntropyLoss()(pred,y)
        optimizer.zero_grad()
        loss.backward()  #反向传播计算梯度
        optimizer.step() #使用梯度下降法更新参数，由于一个batch更新一次，所以采用的整体方法是随机梯度下降算法
test(test_dataloader,model,classifier)

