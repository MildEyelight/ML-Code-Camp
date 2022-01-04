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

#Model
##GNN的block模型，这个模型基本和week3的一致
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

#全图的随机采样模型
##这个模型突破了邻域采样的限制
##任务就是希望突破邻域限制进行采样，在全图上进行随机采样。即使中心节点与采样出的节点之间没有连边
##这样的采样方法也可以将原来的图看作是完全图，以使得GNN能够聚合到不是邻居的节点
class MultiLayerRandomSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts):
        super().__init__(len(fanouts))
        self.fanouts = fanouts

    def sample_frontier(self, block_idx, g, seed_nodes, *args, **kwargs):
        fanout = self.fanouts[block_idx]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            n_nodes = g.number_of_nodes()                 
            if isinstance(seed_nodes, dict):
                n_dst = seed_nodes['_N'].shape[0]
                seed_nodes = seed_nodes['_N']
            else:
                n_dst = seed_nodes.shape[0]
            src = torch.randint(0,n_nodes,size=(int(fanout*n_dst),))
            dst = seed_nodes.repeat(fanout)
            frontier = dgl.graph((src,dst), num_nodes=n_nodes)
        return frontier

    def __len__(self):
        return self.num_layers


#train 过程
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
GNN with stochastic sampling in full graph.
In epoch 295
val  acc: 0.092 (best 0.112) 
test acc: 0.083 (best 0.098)
由于全图的随机采样放弃了GNN聚合邻居的特点和同构性的假设，所以单纯的使用随机采样的方法并不能得到很好的效果。
尤其随机采样甚至会将与该点差别很大的点的feature也聚合进该节点的特征，也就是我觉得可能会像深层GNN那样出现
节点的特征过平滑的现象，每个节点最终的表征都趋于一致。
能否在邻居节点采样的基础上增加一些全图的随机信息来既利用GNN的邻居聚合特点，又使用全图的采样特点。或者感觉
随机采样不能够太随机，可以使用一种概率和链路预测的方法来评估没有连边的点之间的相似程度然后软采样这样子。
这样的话感觉也可能实现全图的采样不过计算成本可能很高。
'''
