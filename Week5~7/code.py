import dgl
import dgl.data
from dgl.nn import GraphConv

import torch
import torch.nn as nn
import torch.nn.functional

import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
#模型使用GPU进行训练
device = "cpu"

#导入数据集并增加self-loop，以及数据集图数据结构的使用。
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
#--Graph数据结构包含了图的node(vertex),edge以及他们的feature和label。
#Graph默认图数据结构是一个有向图，而feature和label并不是必须的。
#下面是几种常见图的分类：
# Undirected Graph: 在dgl可以用bidirectional graph 来表示
# Attribute Graph: node含有feature的一类图,在dgl中这个feature可以是任意阶张量。
# Weighted Graph: edge有权重的图
#--API 展示--#
g.ndata['label']
g.ndata['feat']
g.edata
g = g.to(device)
##搭建有两层隐藏层的GCN模型
##input(g,feature)-->GraphConv1-->GraphConv2-->output
class GCN(nn.Module):
    def __init__(self,in_features,out_features,num_classes,dropout_rate):
        super(GCN, self).__init__()
        self.gconv1 = GraphConv(in_features,out_features)
        self.gconv2 = GraphConv(out_features,num_classes)
        #self.gconv3 = GraphConv(out_features,out_features)
        self.batchnorm = torch.nn.BatchNorm1d(in_features)
        self.dropout_rate = dropout_rate
    def forward(self,g,in_feat):
        #in_feat = self.batchnorm(in_feat)
        hidden_feature = self.gconv1(g,in_feat)
        hidden_feature = torch.nn.functional.relu(hidden_feature)
        hidden_feature = torch.nn.functional.dropout(hidden_feature,self.dropout_rate)
        #hidden_feature = self.batchnorm(hidden_feature)
        #hidden_feature = self.gconv3(g,hidden_feature)
        #hidden_feature = torch.nn.functional.relu(hidden_feature)
        out_feature = self.gconv2(g,hidden_feature)
        #out_feature = self.batchnorm(out_feature)
        return out_feature
##训练GCN模型
def train(graph_data,model,optimizer,epoch=200):
    best_val_acc = 0
    best_test_acc = 0
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for a in range(epoch):
        
        logits = model(g, features)
        pred = logits.argmax(1)

        #只能计算有标签数据(训练数据)的损失，这即是半监督的地方，如果全部节点标签都参与的话就是有监督学习了。
        loss = torch.nn.functional.cross_entropy(logits[train_mask], labels[train_mask])
        
        # 计算训练集,验证集,测试集的准确率
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # 保存训练中最佳的准确率
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if a % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
                a, loss, val_acc, best_val_acc, test_acc, best_test_acc))
    return val_acc

def objective(trial):
    pram = {
        "hidden_feature" : trial.suggest_int("hidden_feature",10,30,step=2),
        "learning_rate" : trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "dropout_rate" : trial.suggest_float("dropout_rate",0,1)
    }
    model = GCN(g.ndata['feat'].shape[1],pram['hidden_feature'],dataset.num_classes,pram['dropout_rate'])
    optimizer = torch.optim.Adam(model.parameters(),lr=pram["learning_rate"])
    model = model.to(device)
    return train(g,model,optimizer)

study = optuna.create_study(direction="maximize",sampler=optuna.samplers.RandomSampler())
n_trials = 15
study.optimize(objective,n_trials)
a = plot_optimization_history(study)
a.show()
b = plot_param_importances(study)
b.show()
c = plot_intermediate_values(study)
c.show()
'''
Result:
In epoch 95
loss: 0.105
val  acc: 0.770 (best 0.780) 
test acc: 0.779 (best 0.761)
'''

'''
Optuna optimization framework citation:
Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta,and Masanori Koyama. 2019.
Optuna: A Next-generation Hyperparameter Optimization Framework. In KDD.
'''
