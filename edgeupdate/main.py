def warn(*args, **kwargs):
    pass

import os
import warnings
warnings.warn = warn
from sklearn import preprocessing,manifold
from arguments import arg_parse
from evaluate_embedding import evaluate_embedding
import random
import networkx as nx
from losses import local_global_loss_
from model import GINEncoder,EDGEUpdateEncoder,PriorDiscriminator,DGCNNModel,GCN,GcnHpoolEncoder
from torch import optim
from torch.autograd import Variable
from torch_geometric.utils import to_networkx,degree,to_dense_batch,to_dense_adj
from torch_geometric.data import InMemoryDataset,Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import json
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from ogb.graphproppred import PygGraphPropPredDataset
import pickle as pkl
from torch_geometric.transforms import BaseTransform
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
from joblib import Parallel, delayed
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from graph2vec import feature_extractor
from rgmvec import EmbeddingMethod,get_emb_transductive,get_wl_labels,rgm
import yaml
from scipy import sparse
from torch.optim.lr_scheduler import ExponentialLR
import seaborn as sns
sns.set(style='darkgrid')
class MyOwnDataset(InMemoryDataset):#我这种是统一处理
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
                #print(data)
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
class Complete(BaseTransform):#这种是顺序处理，但是对于获取节点的度值不够灵活，
    def __init__(self,root):
        with open(root,'rb') as f:
            self.one_hot_dict = pkl.load(f)
        self.num_cla = len(self.one_hot_dict)
    def __call__(self,data):
        #data.to('cuda:0')
        #one_hot_dict = defaultdict(lambda: 'N/A')
        #nlabel_set = set([])
        #for data in self.dataset_a:
        #    nlabel_set = nlabel_set.union(set(int(n1.item()) for n1 in degree(data.edge_index[0],num_nodes=data.num_nodes)))#度值是不连续的，
        #nlabel_set = sorted(list(nlabel_set))
        #for i,node_degree in enumerate(nlabel_set):
        #    one_hot_dict[node_degree] = i
        cow,row = data.edge_index
        deg = degree(cow,data.num_nodes)
        #print(deg)
        deg = torch.tensor([self.one_hot_dict[i.item()] for i in deg],dtype=torch.long)
        deg = F.one_hot(deg, num_classes=self.num_cla).to(torch.float)

        cow_feature = torch.index_select(deg,0,cow)
        row_feature = torch.index_select(deg,0,row)
        edge_feature = cow_feature+row_feature
        data.x = deg[:,:10]
        data.edge_attr = edge_feature[:,:10]
        return data

class InfoGraph(nn.Module):
  def __init__(self, args,input_dim,input_dim_edge, gamma=0.1):#默认hidden_dim维度为32，num_gc_layers为5层
    super(InfoGraph, self).__init__()
    self.gamma = gamma#折中系数
    self.prior = args.prior#损失函数补偿
    self.measure = args.measure#对比损失函数形式
    #print(input_dim)
    #print(hidden_dim)
    #print(output_dim)
    #print(num_gc_layers)
    #print(learn_eps)
    #print(graph_pooling_type)
    #print(neighbor_pooling_type)
    self.embedding_dim = args.hidden_dim * args.num_gc_layers#32×5=160
    if args.model_name == 'gin':
        self.encoder = GINEncoder(input_dim, hidden_dim=args.hidden_dim, output_dim=args.output_dim,num_gc_layers=args.num_gc_layers,embedding_dim=self.embedding_dim,
                                  learn_eps=args.learn_eps,graph_pooling_type=args.graph_pooling_type,neighbor_pooling_type=args.neighbor_pooling_type)
    elif args.model_name == 'edge_gin':
        self.encoder = EDGEUpdateEncoder(input_dim, input_dim_edge,hidden_dim=args.hidden_dim, output_dim=args.output_dim,num_gc_layers=args.num_gc_layers,embedding_dim=self.embedding_dim,
                                          learn_eps=args.learn_eps,graph_pooling_type=args.graph_pooling_type,neighbor_pooling_type=args.neighbor_pooling_type)



    #self.local_d = FF(self.embedding_dim)#只是做了一个全连接层，维度不变，只是参与训练，辅助训练编码器，但是主要还是GINEncoder有用，
    #self.global_d = FF(self.embedding_dim)#只是做了一个全连接层，维度不变


    if self.prior:#为true,执行此句子
        self.prior_d = PriorDiscriminator(self.embedding_dim)#只是参与训练，辅助训练编码器，但是主要还是GINEncoder有用，

    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

  def forward(self, x, edge_index, batch,edge_attr):
    #y, M = self.encoder(x, edge_index, batch,edge_attr)#y返回的是拼接后图的特征，M返回的是拼接后批节点的特征,特征维度为卷积层个数5×输出维度32
    g_enc, l_enc, y = self.encoder(x, edge_index, batch,edge_attr)#g_enc返回的是拼接后图的特征，l_enc返回的是拼接后批节点的特征,特征维度为卷积层个数5×输出维度32
    #x是没有执行mpl的图嵌入，即g_enc的前部
    #g_enc = self.global_d(y)#num_graphs×嵌入维度
    #l_enc = self.local_d(M)#节点个数×嵌入维度
    #print(y.shape)
    #print(M.shape)
    #print(g_enc.shape)
    #print(l_enc.shape)
    #print(l_enc)
    #print(g_enc.t())
    #mode='fd'
    #self.measure='JSD'#默认
    local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, self.measure)#返回的是一个独立的损失值。

    if self.prior:#执行此句
        prior = torch.rand_like(g_enc)#形状一样，里面的值是[0,1)
        #print(prior)
        term_a = torch.log(self.prior_d(prior)).mean()
        #term_b = torch.log(1.0 - self.prior_d(y)).mean()#原语句，但是会出现-inf
        term_b = torch.log(1.0+self.prior_d(y)).mean()
        PRIOR = - (term_a + term_b) * self.gamma#prior返回的也是一个独立的数值。
        #print(term_a)
        #print(term_b)
        #print(PRIOR)
    else:
        PRIOR = 0
    #print(PRIOR)
    #print(local_global_loss + PRIOR)
    return local_global_loss + PRIOR

    

if __name__ == '__main__':
    args = arg_parse()
    torch.manual_seed(30)
    torch.cuda.manual_seed_all(30)
    torch.cuda.manual_seed(30)
    np.random.seed(30)
    random.seed(30)

    batch_size = 128#128个图为一个周期
    lr = args.lr

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'TUDataset')

    for DS in args.DS:
        if DS == 'MUTAG':#原始节点特征，和边特征均存在，节点特征维度7，边特征维度4
            #dataset = TUDataset(path, name='MUTAG')
            dataset_after = TUDataset(path, name='MUTAG')
            hparams = {'channel_list': [7, 30, 30, 30, 30,2],'node_list': [7, 10, 10]}
        elif DS == 'IMDB-BINARY':#原始节点特征、边特征均不存在。构造的节点特征维度65
            dataset = TUDataset(path, name='IMDB-BINARY')
            dataset_after = MyOwnDataset(save_root=(path+'/IMDB-BINARY/after'),dataset=dataset)
            hparams = {'channel_list': [65, 128, 128, 128, 128,2],'node_list': [65, 100, 100]}
            args.hidden_dim = 128
            args.output_dim = 128
        elif DS == 'ENZYMES':#节点特征存在，但是边的原始特征不存在。节点特征维度为3
            dataset = TUDataset(path,name='ENZYMES')
            dataset_after = MyOwnDataset(save_root=(path+'/ENZYMES/after'),dataset=dataset)
            hparams = {'channel_list': [3, 30, 30, 30, 30,6],'node_list': [3, 10, 10]}
        elif DS == 'ogbg-molhiv':
            #dataset  = PygGraphPropPredDataset(root = path,name='ogbg-molhiv')#多增加一句只是为了后续调用方便
            dataset_after  = PygGraphPropPredDataset(root = path,name='ogbg-molhiv')#节点的特征维度为9
            hparams = {'channel_list': [9, 32, 32, 32,32,2],'node_list': [9, 18, 18]}
        elif args.DS == 'COLLAB':
            dataset_after = TUDataset(path,name='COLLAB',pre_transform=Complete(root='model/data/TUDataset/COLLAB/zb.pkl'))
      
        dataloader = DataLoader(dataset_after, batch_size=batch_size)#构造批数据集
        train_loader = DataLoader(dataset_after[:], batch_size=batch_size)
        #print(dataset_after.num_classes)
        print('================')
        print('Dataset name:{}'.format(DS))
        print('lr: {}'.format(lr))#默认学习率0.01
        print('num_node_features: {}'.format(dataset_after.num_features))
        print('num_edge_feature:{}'.format(dataset_after.num_edge_features))
        print('hidden_dim: {}'.format(args.hidden_dim))#隐藏层32维度
        print('out_dim:{}'.format(args.output_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))#5层
        print('eps:{}'.format(args.learn_eps))
        print('has self_loop:{}'.format(dataset_after[0].has_self_loops()))
        print('================')

        if args.train:#训练模型，并保存最佳模型
            if args.model_name == 'node2vec':
                #ret = []
                #y = []
                for i,data in enumerate(dataloader):
                    data.to(device)
                    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20, context_size=10, walks_per_node=20,
                                num_negative_samples=1, p=200, q=1, num_nodes = data.x.shape[0],sparse=True).to(device)
                    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
                    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)#生成正负采样的loader。

                    loss_score = np.inf
                    print(i)
                    for epoch in tqdm(range(args.epochs)):#训练模型
                        loss_all = 0
                        model.train()
                        for pos_rw, neg_rw in loader:
                            #print(pos_rw.shape)
                            #print(neg_rw.shape)
                            optimizer.zero_grad()
                            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                            #print(loss)
                            loss.backward()
                            optimizer.step()
                            loss_all += loss.item()
                        if loss_all < loss_score:
                            #x = global_mean_pool(model.forward(),data.batch)
                            torch.save(model.state_dict(),osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',str(i)+DS+'_'+args.model_name+'_'+'_best_check_model.pt'))
                            loss_score = loss_all
            elif args.model_name == 'graph2vec':
                document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, name,args.wl_iterations) for name,g in tqdm(enumerate(dataset_after)))
                #print(document_collections)
                model = Doc2Vec(document_collections,vector_size=args.dimensions,window=0,
                    min_count=args.min_count,dm=0,sample=args.down_sampling,
                    workers=args.workers,epochs=args.epochs,alpha=args.learning_rate)
                print(len(dataset_after))
                out = []
                for f in range(len(dataset_after)):
                    identifier = str(f)
                    out.append([f] + list(model.docvecs["g_"+identifier])+[dataset_after[f].y.item()])
                column_names = ["type"]+["x_"+str(dim) for dim in range(args.dimensions)]+['y']
                df = pd.DataFrame(out, columns=column_names)
                df = df.sort_values(["type"])
                df.to_csv(osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',DS+'_'+args.model_name+'_'+'_best_check_model.csv'),index=None)
            elif args.model_name == 'rgm':
                args.dimensionality = 6
                emb_method = EmbeddingMethod(method='eigenvector', dimensionality= args.dimensionality, max_layer=2, normalize=False, abs_val=False)
                embs,y = get_emb_transductive(emb_method, graphs = dataset_after)#列表的形式，存储每一个子图的节点的嵌入。此时嵌入的维度还是6
                if args.wliter > 0:#wl测试的次数，默认为2.
                    combined_node_labels, combined_mapping = get_wl_labels(dataset_after, args)#一次性传入所有图。
                    #print(len(combined_node_labels[0]))#是一个列表，存储每一个子图每个节点的标签
                    features = []
                    #print(combined_mapping)
                    for i in range(args.wliter + 1):
                        print("Features from labels at WL iter %d" % i)
                        graph_emb = rgm(embs, args, labels = combined_node_labels[i])
                        print(graph_emb.shape)
                        features.append(graph_emb)
		            #Concatenate feature maps across labelings
                    if sparse.issparse(features[-1]):
                        features = sparse.hstack(features, format = "csr")
                    else:
                        features = np.hstack(features)
                else:
		            #Get unlabeled RGM features
                    features = rgm(embs, args)
                column_names = ["x_"+str(dim) for dim in range(features.shape[1])]
                print(features.shape)
                #print(column_names)
                df = pd.DataFrame(features.toarray(), columns=column_names)
                df['y'] = y
                df.to_csv(osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',DS+'_'+args.model_name+'_'+'_best_check_model.csv'),index=None)
            elif args.model_name == 'xnetmf':
                emb_method = EmbeddingMethod(method=args.emb, max_layer=2, dimensionality= 100, normalize=True, abs_val=True)
                embs,y = get_emb_transductive(emb_method, graphs = dataset_after)#embs:
                column_names = ["x_"+str(dim) for dim in range(embs.shape[1])]
                df = pd.DataFrame(embs.numpy(), columns=column_names)
                df['y'] = y
                df.to_csv(osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',DS+'_'+args.model_name+'_'+'_best_check_model.csv'),index=None)
            elif args.model_name == 'edge_gin' or args.model_name == 'gin':
                model = InfoGraph(args,dataset_after.num_features,dataset_after.num_edge_features).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=10, min_lr=0.000001)

                loss_score = np.inf
                for epoch in range(args.epochs):#训练模型
                    loss_all = 0
                    model.train()
                    for data in dataloader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        loss = model(data.x, data.edge_index, data.batch,data.edge_attr)
                        loss_all += loss.item() * data.num_graphs
                        print(loss)
                        loss.backward()
                        optimizer.step()
                    scheduler.step(loss_all)
                    print('===== Dataset name: {},Epoch: {}, Loss: {} ====='.format(DS,epoch, loss_all / len(dataloader)))#损失函数过大，数据特征能否对劳，下游任务的处理
                    print(scheduler.optimizer.param_groups[0]['lr'])
                    if loss_all < loss_score:
                        torch.save(model.state_dict(),osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',DS+'_'+args.model_name+'_'+args.graph_pooling_type+'_best_check_model.pt'))
                        loss_score = loss_all
        else:
            if args.model_name == 'no_graph':
                ret = []
                y = []
                for data in dataloader:
                    data.to(device)
                    x = global_mean_pool(data.x,data.batch)
                    ret.append(x)
                    y.append(data.y)
                emb = torch.cat(ret,0).float()
                y = torch.cat(y, 0).float()
                if args.vis==True:
                    data_batch = Batch.from_data_list(dataset_after)
                    emb_original = global_mean_pool(data_batch.x,data_batch.batch)
                    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                    x_original = tsne.fit_transform(emb_original.cpu().numpy())

                    df1 = pd.DataFrame(x_original,columns=['x','y'])
                    df1['label'] = data_batch.y.squeeze().cpu().numpy()
                    sns.relplot(x='x',y='y', hue='label',style='label', data=df1)
                    plt.savefig(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_original_'+args.model_name+'_'+'.pdf'))
                    plt.show()
                else:
                    #print(emb.type())#分子数据集是long
                    #print(y.type())
                    res = evaluate_embedding(emb, y)
                    df = pd.DataFrame(res,index=['logreg','svc','beiyes','randomforest'],columns=range(10))#独立重复10次实验 
                    print(df)
                    df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.csv'))
            elif args.model_name == 'node2vec':
                ret = []
                y = []
                for i,data in enumerate(dataloader):
                    data.to(device)
                    best_model_dict = torch.load(osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',str(i)+DS+'_'+args.model_name+'_'+'_best_check_model.pt'),map_location=device)
                    x = global_mean_pool(best_model_dict['embedding.weight'],data.batch)
                    ret.append(x)
                    y.append(data.y)
                emb = torch.cat(ret,0)
                y = torch.cat(y, 0)
                if args.vis==True:
                    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                    X_tsne = tsne.fit_transform(emb.cpu().numpy())

                    df_emb = pd.DataFrame(X_tsne,columns=['x','y'])#独立重复10次实验 
                    df_emb['label'] = y.cpu().squeeze().numpy()

                    sns.relplot(x='x',y='y', hue='label',style='label', data=df_emb)
                    plt.savefig(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.pdf'))
                    plt.show()
                else:
                    res = evaluate_embedding(emb, y)
                    df = pd.DataFrame(res,index=['logreg','svc','beiyes','randomforest'],columns=range(10))#独立重复10次实验 
                    print(df)
                    df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.csv'))
            elif args.model_name == 'graph2vec':
                df = pd.read_csv(osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',DS+'_'+args.model_name+'_'+'_best_check_model.csv'),index_col=0)
                y = df['y'].to_numpy()
                emb = df.iloc[:,:-1].to_numpy()

                if args.vis==True:
                    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                    X_tsne = tsne.fit_transform(emb)

                    df_emb = pd.DataFrame(X_tsne,columns=['x','y'])#独立重复10次实验 
                    df_emb['label'] = y

                    sns.relplot(x='x',y='y', hue='label',style='label', data=df_emb)
                    plt.savefig(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.pdf'))
                    plt.show()
                else:
                    res = evaluate_embedding(emb, y)
                    df = pd.DataFrame(res,index=['logreg','svc','beiyes','randomforest'],columns=range(10))#独立重复10次实验 
                    print(df)
                    df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.csv'))
            elif args.model_name == 'rgm':
                df = pd.read_csv(osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',DS+'_'+args.model_name+'_'+'_best_check_model.csv'),index_col=0)
                y = df['y'].to_numpy()
                emb = df.iloc[:,:-1].to_numpy()

                if args.vis==True:
                    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                    X_tsne = tsne.fit_transform(emb)

                    df_emb = pd.DataFrame(X_tsne,columns=['x','y'])#独立重复10次实验 
                    df_emb['label'] = y

                    sns.relplot(x='x',y='y', hue='label',style='label', data=df_emb)
                    plt.savefig(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.pdf'))
                    plt.show()
                else:
                    res = evaluate_embedding(emb, y)
                    df = pd.DataFrame(res,index=['logreg','svc','beiyes','randomforest'],columns=range(10))#独立重复10次实验 
                    print(df)
                    df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.csv'))
            elif args.model_name == 'xnetmf':
                df = pd.read_csv(osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',DS+'_'+args.model_name+'_'+'_best_check_model.csv'),index_col=0)
                y = df['y'].to_numpy()
                emb = df.iloc[:,:-1].to_numpy()
                if args.vis==True:
                    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                    X_tsne = tsne.fit_transform(emb)

                    df_emb = pd.DataFrame(X_tsne,columns=['x','y'])#独立重复10次实验 
                    df_emb['label'] = y

                    sns.relplot(x='x',y='y', hue='label',style='label', data=df_emb)
                    plt.savefig(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.pdf'))
                    plt.show()
                else:
                    res = evaluate_embedding(emb, y)
                    df = pd.DataFrame(res,index=['logreg','svc','beiyes','randomforest'],columns=range(10))#独立重复10次实验 
                    print(df)
                    df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.csv'))
            elif args.model_name == 'dgcnn':
                train_index = int(len(dataset_after) * 0.2)
                acc = []
                for _ in range(10):
                    train_loader = DataLoader(dataset_after[:train_index], batch_size=batch_size,shuffle=True)
                    test_loader = DataLoader(dataset_after, batch_size=batch_size)
                    #print(dataset_after.num_features)
                    model = DGCNNModel(dataset_after.num_features,dataset_after.num_classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    scheduler = ExponentialLR(optimizer, gamma=0.9)  
                    criterion = torch.nn.CrossEntropyLoss()
                
                    for epoch in range(args.epochs):#训练模型
                        model.train()
                        for data in train_loader:#所有数据参与训练
                            data = data.to(device)
                            optimizer.zero_grad()
                            out = model(data.x, data.edge_index, data.batch,)
                            loss = criterion(out,data.y.squeeze())
                            loss.backward()
                            optimizer.step()

                    model.eval()
                    correct = 0
                    for data in test_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.batch)
                        out = torch.softmax(out,dim=1)
                        pred = out.argmax(dim=1)
                        correct += int((pred == data.y.squeeze()).sum())
                    acc.append(correct / len(dataloader.dataset))
                df = pd.DataFrame(acc,columns=range(1))#独立重复10次实验 
                print(df)
                print(df[0].mean())
                print(df[0].std())
                df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.csv'))
            elif args.model_name == 'gcn':
                train_index = int(len(dataset_after) * 0.2)
                acc = []
                for _ in range(10):
                    train_loader = DataLoader(dataset_after[:train_index], batch_size=batch_size,shuffle=True)
                    test_loader = DataLoader(dataset_after, batch_size=batch_size)
                    model = GCN(dataset_after.num_features,args.hidden_dim,dataset_after.num_classes).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    scheduler = ExponentialLR(optimizer, gamma=0.9)  
                    criterion = torch.nn.CrossEntropyLoss()

                    for epoch in range(args.epochs):#训练模型
                        model.train()
                        for data in train_loader:
                            data = data.to(device)
                            optimizer.zero_grad()
                            out = model(data.x, data.edge_index, data.batch)#x特征必须是float类型，edge_index是long，
                            loss = criterion(out,data.y.squeeze())
                            loss.backward()
                            optimizer.step()
                            scheduler.step()  

                    model.eval()
                    correct = 0
                    for data in test_loader:
                        data = data.to(device)
                        out = model(data.x, data.edge_index, data.batch)
                        out = torch.softmax(out,dim=1)
                        pred = out.argmax(dim=1)
                        correct += int((pred == data.y.squeeze()).sum())
                    acc.append(correct / len(test_loader.dataset))
                df = pd.DataFrame(acc,columns=range(1))#独立重复10次实验 
                print(df)
                print(df[0].mean())
                print(df[0].std())
                df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.csv'))
            elif args.model_name == 'diffpool':
                train_index = int(len(dataset_after) * 0.4)
                acc = []
                for _ in range(10):
                    dataset_after = dataset_after.shuffle()
                    train_loader = DataLoader(dataset_after[:train_index], batch_size=batch_size)
                    model = GcnHpoolEncoder(hparams).to(device)
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    for epoch in tqdm(range(args.epochs)):
                        model.train()
                        for batch_idex, graph_data in enumerate(train_loader):
                            optimizer.zero_grad()
                            ypred = model(graph_data.to(device))
                            #print(ypred.shape)
                            #print(graph_data.y.squeeze().shape)
                            loss = criterion(ypred, graph_data.y.squeeze())
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                            optimizer.step()

                    model.eval()
                    correct = 0
                    for data in dataloader:
                        data = data.to(device)
                        out = model(data)
                        out = torch.softmax(out,dim=1)
                        pred = out.argmax(dim=1)
                        print(pred.shape)
                        correct += int((pred == data.y.squeeze()).sum())
                    acc.append(correct / len(dataloader.dataset))
                df = pd.DataFrame(acc,columns=range(1))#独立重复10次实验 
                print(df)
                df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+'.csv'))
            elif args.model_name == 'efruge_end_to_end':#dataset_after.num_features,dataset_after.num_edge_features

                train_index = int(len(dataset_after) * 0.2)
                print(train_index)
                acc = []
                for _ in range(10):
                    #dataset_after = dataset_after
                    train_loader = DataLoader(dataset_after[:train_index], batch_size=batch_size,shuffle=True)
                    test_loader = DataLoader(dataset_after, batch_size=batch_size)
                    model = EDGEUpdateEncoder(dataset_after.num_features,dataset_after.num_edge_features,
                                              args.hidden_dim,args.output_dim,args.num_gc_layers,
                                              args.hidden_dim * args.num_gc_layers,args.learn_eps,args.graph_pooling_type,args.neighbor_pooling_type).to(device)
                    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                    scheduler = ExponentialLR(optimizer, gamma=0.9)  
                    criterion = torch.nn.CrossEntropyLoss()
                    
                    for epoch in tqdm(range(args.epochs)):
                        model.train()
                        for _,graph_data in enumerate(train_loader):
                            graph_data = graph_data.to(device)
                            optimizer.zero_grad()
                            ypred,_,_ = model(graph_data.x,graph_data.edge_index,graph_data.batch,graph_data.edge_attr)
                            #print(ypred.shape)
                            #print(graph_data.y.squeeze().shape)#维度要比ypred少一维才行。
                            loss = criterion(ypred, graph_data.y.squeeze())#loss 反向传播前后的输出值是一样的。
                            loss.backward()
                            #torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                            optimizer.step()
                            scheduler.step()  
                    model.eval()
                    correct = 0
                    for data in test_loader:
                        data = data.to(device)
                        out,_,_ = model(data.x,data.edge_index,data.batch,data.edge_attr)
                        out = torch.softmax(out,dim=1)
                        pred = out.argmax(dim=1)
                        #print(pred.shape)
                        correct += int((pred == data.y.squeeze()).sum())
                    acc.append(correct / len(test_loader.dataset))
                df = pd.DataFrame(acc,columns=range(1))#独立重复10次实验 
                print(df)
                print(df[0].mean())
                print(df[0].std())
                df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+args.graph_pooling_type+'_'+'.csv'))
            elif args.model_name == 'edge_gin':
                model = InfoGraph(args,dataset_after.num_features,dataset_after.num_edge_features).to(device)
                best_model_dict = torch.load(osp.join('/home/pst/myc/zb1/PythonApplication4/model/best_model',DS+'_'+args.model_name+'_'+args.graph_pooling_type+'_best_check_model.pt'),map_location=device)
                model_state_dict = model.state_dict()
                model_state_dict.update(best_model_dict)
                model.load_state_dict(model_state_dict)
                model.eval()
                with torch.no_grad():
                    emb, y = model.encoder.get_embeddings(dataloader)
                if args.vis == True:
                    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                    X_tsne = tsne.fit_transform(emb.cpu().numpy())
                   
                    df = pd.DataFrame(X_tsne,columns=['x','y'])#独立重复10次实验 
                    df['label'] = y.cpu().numpy()

                    sns.relplot(x='x',y='y', hue='label',style='label', data=df)
                    plt.savefig(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+args.graph_pooling_type+'_'+'.pdf'))
                    plt.show()
                else:
                    res = evaluate_embedding(emb, y)
                    df = pd.DataFrame(res,index=['logreg','svc','beiyes','randomforest'],columns=range(10))#独立重复10次实验 
                    print(df)
                    df.to_csv(osp.join(osp.dirname(osp.realpath(__file__)), 'result','trained_'+DS+'_'+args.model_name+'_'+args.graph_pooling_type+'.csv'))

