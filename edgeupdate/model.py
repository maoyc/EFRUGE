from torch import optim
from torch.autograd import Variable
import json
import numpy as np
import torch.nn as nn
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC, LinearSVC
from torch.nn import Sequential, Linear, ReLU,BatchNorm1d,Conv1d, MaxPool1d, Dropout
#from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GINConv, global_add_pool,global_mean_pool,global_max_pool,MessagePassing,global_sort_pool,GCNConv
from tqdm import tqdm
import os.path as osp
import sys
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset
from collections import defaultdict
from torch_geometric.utils import to_networkx,degree,dense_to_sparse,remove_self_loops,to_dense_batch,to_dense_adj
from torch.nn.parameter import Parameter
import math


class EdgeupdateConv(MessagePassing):
    def __init__(self,in_channels_node,in_channels_edge,out_channels,learn_eps,aggr):
        super().__init__(aggr=aggr)#邻居节点和连边特征聚合
        self.mlp_node = torch.nn.Sequential(torch.nn.Linear(in_channels_node, 2*in_channels_node), torch.nn.BatchNorm1d(2*in_channels_node), torch.nn.ReLU(), torch.nn.Linear(2*in_channels_node, out_channels))
        self.eps = torch.nn.Parameter(torch.tensor(learn_eps))
        self.mlp_edge = torch.nn.Sequential(torch.nn.Linear(2*in_channels_node+in_channels_edge, 2*in_channels_edge), torch.nn.BatchNorm1d(2*in_channels_edge), torch.nn.ReLU(), torch.nn.Linear(2*in_channels_edge, out_channels))
        self.mlp_combin = torch.nn.Linear(in_channels_node+in_channels_edge,in_channels_node)
    def forward(self,x,edge_index,edge_attr):#propogate返回的是节点的特征，前提是边的特征维度和节点特征维度不一致的问题。
        #print(x.device)
        #print(edge_index.device)
        #print(edge_attr.device)
        out = self.mlp_node((1+self.eps)*x+self.propagate(edge_index,x=x,edge_attr=edge_attr))
        row,col = edge_index
        row_embedding = torch.index_select(x,0,row)
        col_embedding = torch.index_select(x,0,col)
        edge_embedding_cat = torch.cat((row_embedding,col_embedding,edge_attr),dim=1)
        edge_embedding = self.mlp_edge(edge_embedding_cat)
        return out,edge_embedding
    def message(self, x_j,edge_attr):#源节点特征加上边的特征,消息的汇聚方式，将所有源节点特征和便特征相加
        #print(x_j.device)
        #print(edge_attr.device)
        a = x_j.shape[1]
        b = edge_attr.shape[1]
        c = torch.cat((x_j,edge_attr),dim=1)
        #print(a)
        #print(b)
        #print(c.device)
        c = self.mlp_combin(c)
        #print(c.device)
        return F.relu(c)

class EDGEUpdateEncoder(torch.nn.Module):
    def __init__(self,input_dim_node,input_dim_edge,hidden_dim,output_dim,num_gc_layers,embedding_dim,learn_eps,graph_pooling_type,neighbor_pooling_type):
        super(EDGEUpdateEncoder,self).__init__()

        self.num_gc_layers = num_gc_layers
        self.learn_eps = learn_eps
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.embedding_dim = embedding_dim
        self.local_d = FF(self.embedding_dim)#只是做了一个全连接层，维度不变，只是参与训练，辅助训练编码器，但是主要还是GINEncoder有用，
        self.global_d = FF(self.embedding_dim)

        if neighbor_pooling_type == 'sum':
            aggr = 'add'
        elif neighbor_pooling_type == 'mean':
            aggr = 'mean'
        elif neighbor_pooling_type == 'max':
            aggr = 'max'

        for i in range(self.num_gc_layers):
            if i:
                self.convs.append(EdgeupdateConv(hidden_dim,hidden_dim,hidden_dim,self.learn_eps,aggr))
            else:#第一层执行此句
                self.convs.append(EdgeupdateConv(input_dim_node,input_dim_edge,hidden_dim,self.learn_eps,aggr))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))#还是需要对特征做个归一化处理
        
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(self.num_gc_layers):
            self.linears_prediction.append(torch.nn.Linear(hidden_dim, output_dim))
        self.drop = torch.nn.Dropout(0.5)

        if graph_pooling_type == 'sum':
            self.pool = global_add_pool
        elif graph_pooling_type == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling_type == 'max':
            self.pool = global_max_pool
        else:
            raise NotImplementedError
            
    def forward(self,x,edge_index,batch,edge_attr):
        xs = []
        if isinstance(x,torch.cuda.LongTensor):
            x = x.float()
        #print(x.device)#,传进来的数据在GPU上
        for i in range(self.num_gc_layers):
            x,edge_attr = self.convs[i](x,edge_index,edge_attr)
            x = self.bns[i](x)
            edge_attr = self.bns[i](edge_attr)
            x = F.relu(x)
            xs.append(x)
        xpool = []
        for i,h in enumerate(xs):
            pooled_h = self.pool(h,batch)
            xpool.append(self.drop(self.linears_prediction[i](pooled_h)))
        x = torch.cat(xpool,1)
        #y = torch.cat(xs,1)
        graph_embedding = self.global_d(x)
        node_embedding = self.local_d(torch.cat(xs,1))
        #return x, torch.cat(xs,1)
        return graph_embedding,node_embedding,x
    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch,data.edge_attr
                x, _, _ = self.forward(x, edge_index, batch,edge_attr)
                ret.append(x)
                y.append(data.y)
        ret = torch.cat(ret, 0)#将所有的图嵌入进行拼接
        y = torch.cat(y, 0)
        y.squeeze_()
        return ret, y#ret是每个图的嵌入，y=[],存储的是每一个图的标签，没有维度

class GINEncoder(torch.nn.Module):
    def __init__(self, input_dim,hidden_dim,output_dim,num_gc_layers,embedding_dim,learn_eps,graph_pooling_type,neighbor_pooling_type):
        super(GINEncoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        '''
        print(input_dim)369
        print(hidden_dim)32
        print(output_dim)32
        print(num_gc_layers)5
        print(learn_eps)0.0
        print(graph_pooling_type)sum
        print(neighbor_pooling_type)sum
        '''
        self.num_gc_layers = num_gc_layers
        self.learn_eps = learn_eps
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.embedding_dim = embedding_dim
        self.local_d = FF(self.embedding_dim)#只是做了一个全连接层，维度不变，只是参与训练，辅助训练编码器，但是主要还是GINEncoder有用，
        self.global_d = FF(self.embedding_dim)

        if neighbor_pooling_type == 'sum':
            aggr = 'add'
        elif neighbor_pooling_type == 'mean':
            aggr = 'mean'
        elif neighbor_pooling_type == 'max':
            aggr = 'max'

        for i in range(self.num_gc_layers):
            if i:
                nn = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))#这里的感知机篇简单了
            else:#第一层执行此句
                nn = Sequential(Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nn,self.learn_eps,True,aggr=aggr))
            self.bns.append(torch.nn.BatchNorm1d(hidden_dim))

        #将图特征做一个转换，
        self.linears_prediction = torch.nn.ModuleList()
        for layer in range(num_gc_layers):
            self.linears_prediction.append(torch.nn.Linear(hidden_dim, output_dim))
        self.drop = torch.nn.Dropout(0.5)

        if graph_pooling_type == 'sum':
            self.pool = global_add_pool
        elif graph_pooling_type == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling_type == 'max':
            self.pool = global_max_pool
        else:
            raise NotImplementedError


    def forward(self, x, edge_index, batch,edge_attr = None):#不传递边的特征
        xs = []#每一层的节点嵌入,
        if isinstance(x,torch.cuda.LongTensor):
            x = x.float()
        for i in range(self.num_gc_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)#batch normy要再激活函数之前引用，特征进行归一化操作
            x = F.relu(x)
            xs.append(x)#每一层节点的嵌入


        xpool = []#对每一层的图嵌入，
        for i, h in enumerate(xs):
            pooled_h = self.pool(h, batch)#一次得到每一层的图嵌入
            xpool.append(self.drop(self.linears_prediction[i](pooled_h)))#将每一层的图嵌入图做预测操作
        x = torch.cat(xpool,1)#此时的x还是比较小的
        #y = torch.cat(xs,1)
        #print(x[:,0])
        graph_embedding = self.global_d(x)
        node_embedding = self.local_d(torch.cat(xs,1))
        #print(graph_embedding[:,0])
        return graph_embedding, node_embedding,x#1维的方向为：层数×输出的维度

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                x, _, _ = self.forward(x, edge_index, batch)
                ret.append(x)
                y.append(data.y)
        ret = torch.cat(ret, 0)#将所有的图嵌入进行拼接
        y = torch.cat(y, 0)
        y.squeeze_()
        #print(ret[1058,0])
        return ret, y#ret是每个图的嵌入，y=[],存储的是每一个图的标签，没有维度

class DGCNNModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DGCNNModel, self).__init__()

        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.conv4 = GCNConv(32, 1)
        self.conv5 = Conv1d(1, 16, 97, 97)
        self.conv6 = Conv1d(16, 32, 5, 1)
        self.pool = MaxPool1d(2, 2)
        self.classifier_1 = Linear(352, 128)
        self.drop_out =torch.nn.Dropout(0.5)
        self.classifier_2 = Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x,edge_index,batch):
        #x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = remove_self_loops(edge_index)
        #print(x.shape)
        x = x.float()
        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        #print(x_4.shape)
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        #print(x.shape)
        x = global_sort_pool(x, batch, k=30)
        #print(x.shape)
        x = x.view(x.size(0), 1, x.size(-1))
        #print(x.shape)
        x = self.relu(self.conv5(x))
        #print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        x = self.relu(self.conv6(x))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        out = self.relu(self.classifier_1(x))
        out = self.drop_out(out)#并不改变维度，只是以一定的概率将一些元素置为0。
        #classes = F.log_softmax(self.classifier_2(out), dim=-1)
        classes = self.classifier_2(out)#因为最外层的损失函数是交叉熵
        return classes

class GCN(torch.nn.Module):
    def __init__(self, num_features,hidden_channels,num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = x.float()
        print(x.type())
        print(edge_index.type())
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
class GraphConvolution(torch.nn.Module):
    def __init__(self, in_features, out_features,bias=True):
        super(GraphConvolution, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adj):
        node_num = adj.shape[-1]
        # add remaining self-loops
        self_loop = torch.eye(node_num).to(self.device)
        self_loop = self_loop.reshape((1, node_num, node_num))
        self_loop = self_loop.repeat(adj.shape[0], 1, 1)
        adj_post = adj + self_loop
        # signed adjacent matrix
        deg_abs = torch.sum(torch.abs(adj_post), dim=-1)#存储每一个子图里面每一个节点的度之和，包括自循环
        deg_abs_sqrt = deg_abs.pow(-0.5)
        diag_deg = torch.diag_embed(deg_abs_sqrt, dim1=-2, dim2=-1)#扩充对角矩阵

        norm_adj = torch.matmul(torch.matmul(diag_deg, adj_post), diag_deg)
        return norm_adj

    def forward(self, input, adj):
        #print(input.type())
        support = torch.matmul(input, self.weight)
        adj_norm = self.norm(adj)
        output = torch.matmul(support.transpose(1, 2), adj_norm.transpose(1, 2))
        output = output.transpose(1, 2)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

def dense_diff_pool(x, adj, s, mask=None):#s为原始节点特征
    #print(x.dim())#初始维度均为3
    #print(adj.dim())
    #print(s.dim())
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2)) + 1e-30
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()
    ent_loss = (-s * torch.log(s + 1e-30)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss

class GcnHpoolSubmodel(torch.nn.Module):
    #channel_list: [3, 30, 30, 30, 30],node_list: [3, 10, 10]
    def __init__(self, in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node):#依次为90，30，30，3，10，10
        super(GcnHpoolSubmodel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_graph(in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node)
        self.reset_parameters()
        self.pool_tensor = None

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, GraphConvolution):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

    def build_graph(self, in_feature, hidden_feature, out_feature, in_node, hidden_node, out_node):
        self.embed_conv_first = GraphConvolution(in_features=in_feature,out_features=hidden_feature).to(self.device)
        self.embed_conv_block = GraphConvolution(in_features=hidden_feature,out_features=hidden_feature).to(self.device)
        self.embed_conv_last = GraphConvolution(in_features=hidden_feature,out_features=out_feature).to(self.device)

        self.pool_conv_first = GraphConvolution(in_features=in_node,out_features=hidden_node).to(self.device)
        self.pool_conv_block = GraphConvolution(in_features=hidden_node,out_features=hidden_node).to(self.device)
        self.pool_conv_last = GraphConvolution(in_features=hidden_node,out_features=out_node).to(self.device)
 
        self.pool_linear = torch.nn.Linear(hidden_node * 2 + out_node, out_node)

    def forward(self, embedding_tensor, pool_x_tensor, adj, embedding_mask):

        pooling_tensor = self.gcn_forward(pool_x_tensor, adj,self.pool_conv_first, self.pool_conv_block, self.pool_conv_last,embedding_mask)            
        pooling_tensor = F.softmax(self.pool_linear(pooling_tensor), dim=-1)#softmax不改变维度
        #print(pooling_tensor.shape)
        if embedding_mask is not None:
            pooling_tensor = pooling_tensor * embedding_mask
        #print(pooling_tensor.shape)
        x_pool, adj_pool, _, _ = dense_diff_pool(embedding_tensor, adj, pooling_tensor)
        #print(x_pool.shape)
        #print(adj_pool.shape)
        embedding_tensor = self.gcn_forward(x_pool, adj_pool,self.embed_conv_first, self.embed_conv_block, self.embed_conv_last)
        output, _ = torch.max(embedding_tensor, dim=1)
        self.pool_tensor = pooling_tensor

        return output, adj_pool, x_pool, embedding_tensor

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
        out_all = []

        layer_out_1 = F.relu(conv_first(x, adj))
        layer_out_1 = self.apply_bn(layer_out_1)
        out_all.append(layer_out_1)

        layer_out_2 = F.relu(conv_block(layer_out_1, adj))
        layer_out_2 = self.apply_bn(layer_out_2)
        out_all.append(layer_out_2)

        layer_out_3 = conv_last(layer_out_2, adj)
        out_all.append(layer_out_3)
        out_all = torch.cat(out_all, dim=2)
        if embedding_mask is not None:
            out_all = out_all * embedding_mask

        return out_all

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)

class GcnHpoolEncoder(torch.nn.Module):
    def __init__(self, hparams):
        super(GcnHpoolEncoder, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._hparams = hparams
        self.build_graph()
        self.reset_parameters()


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, GraphConvolution):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

    def build_graph(self):
        '''
        channel_list: [3, 30, 30, 30, 30,6]
        '''
        self.entry_conv_first = GraphConvolution(in_features=self._hparams['channel_list'][0],out_features=self._hparams['channel_list'][1]).to(self.device)
        self.entry_conv_block = GraphConvolution(in_features=self._hparams['channel_list'][1],out_features=self._hparams['channel_list'][1]).to(self.device)
        self.entry_conv_last = GraphConvolution(in_features=self._hparams['channel_list'][1],out_features=self._hparams['channel_list'][2]).to(self.device)
        
        self.gcn_hpool_layer = GcnHpoolSubmodel(self._hparams['channel_list'][2] * 3, self._hparams['channel_list'][3], self._hparams['channel_list'][4],self._hparams['node_list'][0], self._hparams['node_list'][1], self._hparams['node_list'][2]).to(self.device)
        #180转化为30，再转化为30
        self.pred_model = torch.nn.Sequential(torch.nn.Linear(2 * 3 * self._hparams['channel_list'][2], self._hparams['channel_list'][3]),torch.nn.ReLU(),torch.nn.Linear(self._hparams['channel_list'][3], self._hparams['channel_list'][-1])).to(self.device)

    def forward(self, graph_input):
        node_feature,embedding_mask = to_dense_batch(graph_input.x.float(),graph_input.batch)
        adjacency_mat = to_dense_adj(graph_input.edge_index,graph_input.batch)
        embedding_mask = embedding_mask.unsqueeze(2)
        #print(graph_input.x.type())
        # entry embedding gcn
        embedding_tensor_1 = self.gcn_forward(node_feature, adjacency_mat,self.entry_conv_first, self.entry_conv_block, self.entry_conv_last,embedding_mask)
        #print(embedding_tensor_1.shape) #128,100,90
        output_1, _ = torch.max(embedding_tensor_1, dim=1)
        #print(output_1.shape)#128,90
        # hpool layer
        output_2, _, _, _ = self.gcn_hpool_layer(embedding_tensor_1, node_feature, adjacency_mat, embedding_mask)
        
        #print(output_2.shape)#128,90
        output = torch.cat([output_1, output_2], dim=1)
        ypred = self.pred_model(output)#30

        return ypred

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
        out_all = []

        layer_out_1 = F.relu(conv_first(x, adj))
        layer_out_1 = self.apply_bn(layer_out_1)
        out_all.append(layer_out_1)

        layer_out_2 = F.relu(conv_block(layer_out_1, adj))
        layer_out_2 = self.apply_bn(layer_out_2)
        out_all.append(layer_out_2)

        layer_out_3 = conv_last(layer_out_2, adj)
        out_all.append(layer_out_3)
        out_all = torch.cat(out_all, dim=2)#最后一个维度是90，三层卷积层叠加在一起。
        if embedding_mask is not None:
            out_all = out_all * embedding_mask#广播机制

        return out_all

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self.device)
        return bn_module(x)


class PriorDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, input_dim)
        self.l1 = nn.Linear(input_dim, input_dim)
        self.l2 = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))#映射为(-1,1)

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim) #没有用激活函数，可能导致的结果值比较大
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #print(F.relu(self.linear_shortcut(x))[:,0])#训练导致数太大了
        return self.block(x) + self.linear_shortcut(x)

class MyOwnDataset(InMemoryDataset):
    def __init__(self, save_root, dataset,transform=None, pre_transform=None, pre_filter=None,device='cuda:0'):
        self.dataset_a = dataset
        self.device = device
        super(MyOwnDataset,self).__init__(save_root,transform, pre_transform, pre_filter)
        
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['origin_dataset']

    @property
    def processed_file_names(self):
        return 'toy_dataset.pt'

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        if self.dataset_a[0].x == None:#节点特征不存在且边特征也不存在的时候执行此句
            data_list = []
            one_hot_dict = defaultdict(lambda: 'N/A')
            nlabel_set = set([])
            for data in self.dataset_a:
                nlabel_set = nlabel_set.union(set(int(n1.item()) for n1 in degree(data.edge_index[0],num_nodes=data.num_nodes)))#度值是不连续的，
            nlabel_set = sorted(list(nlabel_set))
            for i,node_degree in enumerate(nlabel_set):
                one_hot_dict[node_degree] = i
            for data in self.dataset_a:
                cow,row = data.edge_index
                deg = degree(cow,data.num_nodes)
                #print(deg)
                deg = torch.tensor([one_hot_dict[i.item()] for i in deg],dtype=torch.long)
                deg = F.one_hot(deg, num_classes=len(nlabel_set)).to(torch.float)
                cow_feature = torch.index_select(deg,0,cow)
                row_feature = torch.index_select(deg,0,row)
                edge_feature = cow_feature+row_feature
                data.x = deg
                data.edge_attr = edge_feature
                print(data)
                data.to(self.device)
                data_list.append(data)
            data_save, data_slices = self.collate(data_list)
            torch.save((data_save, data_slices), self.processed_paths[0])
        else:#节点特征存在，但是边的特征不存在的时候
            data_list = []
            for data in self.dataset_a:
                cow,row = data.edge_index

                cow_feature = torch.index_select(data.x,0,cow)
                row_feature = torch.index_select(data.x,0,row)
                edge_feature = cow_feature+row_feature
                data.edge_attr = edge_feature
                data.to(self.device)
                data_list.append(data)
            data_save, data_slices = self.collate(data_list)
            torch.save((data_save, data_slices), self.processed_paths[0])

if __name__=='__main__':
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TUDataset')
    dataset = TUDataset(root=path, name='MUTAG',use_edge_attr=True).shuffle()#COLLAB,ENZYMES,MUTAG,ENZYMES（已处理）数据集的节点特征是有的，
    #但是边的特征不存在,MUTAG（无需处理）边和节点的属性都存在，IMDB-BINARY（已处理）的节点和边特征均不存在。
    #dataset_after = MyOwnDataset(save_root=(path+'/IMDB-BINARY/after'),dataset=dataset)
    print(dataset.data.x.type())
    print(dataset.data.edge_index.type())
    #print(dataset_after)
    #print(dataset_after[0])
    #print(dataset_after[0].num_nodes)
    ##print(dataset[0].x.shape)
    ##print(dataset[0].edge_index.shape)
    ##print(dataset[0].edge_attr.shape)
    #input_dim_node = dataset.num_node_features
    #input_dim_edge = dataset.num_edge_features
    #hidden_dim = 32
    #output_dim = 32
    #num_gc_layers = 5
    #learn_eps = 0.0
    #graph_pooling_type = 'sum'
    #neighbor_pooling_type = 'sum'
    #dataloader = DataLoader(dataset, batch_size=30)
    ###print(dataset.data.x.type())
    ###edge_attr = torch.ones((dataset[0].edge_index.shape[1],dataset[0].x.shape[1]),dtype=torch.float)
    ###dataset[0].edge_attr = edge_attr
    ###print(dataset[0].edge_attr)
    ###print(dataset.data.edge_attr)
    #model_edge =EDGEUpdateEncoder(input_dim_node,input_dim_edge,hidden_dim,output_dim,num_gc_layers,
    #                  learn_eps,graph_pooling_type,neighbor_pooling_type)

    #for i,data in enumerate(dataloader):
    #    y,m = model_edge(data.x,data.edge_index,data.edge_attr,data.batch)
    #    print(data)
    #    print(y.shape)
    #    print(m.shape)
    #    break
    ##a = torch.tensor([[1,2,3],[4,5,6]])
    ##b = torch.tensor([[1,2],[4,5]])
    ##c = torch.cat((a,b),dim=1)
    ##print(c.shape)