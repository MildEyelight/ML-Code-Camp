import dgl
import dgl.nn
import torch
import torch.nn
import torch.nn.functional
from ogb.nodeproppred import DglNodePropPredDataset,Evaluator

device = 'cpu'

#import dataset of obgn_arxiv
dataset = DglNodePropPredDataset('ogbn-arxiv')
g,node_label = dataset[0]
##obgn_arxiv is a dircted graph and we turn it to undircted graph
g = dgl.add_reverse_edges(g)
g = dgl.add_self_loop(g)
g.ndata['label'] = node_label[:, 0]
split_idx = dataset.get_idx_split()
train_idx,valid_idx,test_idx = split_idx["train"],split_idx["valid"],split_idx["test"]
train_mask = torch.zeros(g.ndata['feat'].shape[0])
valid_mask = torch.zeros(g.ndata['feat'].shape[0])
test_mask = torch.zeros(g.ndata['feat'].shape[0])
train_mask[train_idx] = 1
valid_mask[valid_idx] = 1
test_mask[test_idx] = 1
 
#Sampling using fullneiborsampling
##Interface provided by dgl
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
##and use dataloader to load three set of data:train,valid,test
train_dataloader = dgl.dataloading.NodeDataLoader(
    g, train_idx, sampler,
    batch_size=1024,
    device=device,
    shuffle=True,
    drop_last=False,
    num_workers=0)
valid_dataloader = dgl.dataloading.NodeDataLoader(
    g, valid_idx, sampler,
    device=device,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0
)
test_dataloader = dgl.dataloading.NodeDataLoader(
    g, test_idx, sampler,
    device=device,
    batch_size=1024,
    shuffle=False,
    drop_last=False,
    num_workers=0
)

#define Stochastic GNN
class StochasticTwoLayerGCN(torch.nn.Module):
    def __init__(self, in_features_num, hidden_features_num, out_features_num):
        super(StochasticTwoLayerGCN,self).__init__()
        self.relu = torch.nn.functional.relu
        self.conv1 = dgl.nn.GraphConv(in_features_num, hidden_features_num)
        self.conv2 = dgl.nn.GraphConv(hidden_features_num, out_features_num)

    def forward(self, blocks, x):
        x = self.relu(self.conv1(blocks[0],x))
        #out_features = torch.nn.functional.dropout(out_features,0.5)
        x = self.relu(self.conv2(blocks[1],x))
        return x
    
#train and test
def train(g,model,epoch=300):
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-2) 
    for a in range(epoch):
        for input_nodes, output_nodes, blocks in train_dataloader:
            blocks = [b.to(torch.device(device)) for b in blocks]
            input_features = blocks[0].srcdata     # returns a dict
            output_labels = blocks[-1].dstdata     # returns a dict
            pred = model(blocks, input_features)
            loss = torch.nn.functional.cross_entropy(output_labels, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #calculate the accuracy on three dataset.
        #In calculate we pass the parameter through the whole graph instead of through subgraph in block.
        test_block = [g,g]
        pred = model(test_block,g.ndata['label']).argmax(1)
        train_acc = (pred[train_mask] == g.ndata['labels'][train_mask]).float().mean()
        val_acc = (pred[val_mask] == g.ndata['labels'][val_mask]).float().mean()
        test_acc = (pred[test_mask] == g.ndata['labels'][test_mask]).float().mean()
        if a % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} , test acc: {:.3f} '.format(
                a, loss, val_acc, , test_acc, ))
model =  StochasticTwoLayerGCN(g.ndata['feat'].shape[1],100,dataset.num_classes)
model.to(device)
train(g,model)

'''
Result Analysis:
(1)GNN with the graph without selfloop 
In epoch 295
val  acc: 0.670 (best 0.680) 
test acc: 0.661 (best 0.679)

(2)GCN with selfloop
In epoch 295
val  acc: 0.709 (best 0.712) 
test acc: 0.735 (best 0.740)

'''
