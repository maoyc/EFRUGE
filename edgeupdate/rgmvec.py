import numpy as np
from scipy.sparse import linalg as sp_linalg
from torch_geometric.utils import to_scipy_sparse_matrix
import time
import os
from histograms import *
from scipy import sparse
from xnetmf import get_representations as xnetmf_embed
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import torch
class EmbeddingMethod():
	def __init__(self, align_info = None, dimensionality=None, max_layer=None, method="xnetmf", alpha = 0.1, normalize = False, abs_val = False, gammastruc = 1, gammaattr = 1, implicit_factorization = True, landmark_features = None, landmark_indices = None, l2l_decomp = None, use_landmarks = False):
		self.method = method #representation learning method
		self.normalize = normalize
		self.abs_val = abs_val
		self.dimensionality = dimensionality #sample p points
		self.max_layer = max_layer #furthest hop distance up to which to compare neighbors
		self.alpha = alpha #discount factor for higher layers
		self.gammastruc = gammastruc
		self.gammaattr = gammaattr
		self.implicit_factorization = implicit_factorization
		self.landmark_features = landmark_features
		self.landmark_indices = None
		self.l2l_decomp = l2l_decomp
		self.use_landmarks = False #use hard coded landmarks

def get_emb_transductive(emb_method, graphs = None):
	
	graph_labels = graphs.data.y.cpu().numpy()#labels of graphs
	individual = (emb_method.method == "eigenvector")

	'''embed all graphs'''
	before_embed = time.time()
	if emb_method.method == "xnetmf": #only get landmark embs if using xNetMF TODO hacky
		embs= multi_network_embeddings(graphs, emb_method, individual=individual)#当方法为xnetmf的时候，individual为0
	else:
		embs = multi_network_embeddings(graphs, emb_method, individual=individual)
	after_embed = time.time()
	print("learned embeddings in time: ", after_embed - before_embed)	
	#print(len(embs))
	return embs, graph_labels

def multi_network_embeddings(graphs, emb_method, individual = True):
	if individual: #learn embeddings on graphs individually
		embs = [learn_embeddings(graph, emb_method) for graph in graphs]
		return embs
	else:#xnetmf的时候执行此
		#Combine graphs into one big adjacency matrix
		graph_batch = Batch.from_data_list(graphs)#生成一个批次的大图
		combined_embs = learn_embeddings(graph_batch, emb_method)
		combined_embs = torch.tensor(combined_embs).cpu()
		embs = global_mean_pool(combined_embs, graph_batch.batch.cpu())#直接返回图嵌入，利用池化。
		return embs

def learn_embeddings(graph, emb_method):#graph是一个data,py类型的
	method = emb_method.method.lower()
	if method == "xnetmf":
		embeddings = xnetmf_embed(graph, emb_method)
		#print(embeddings)
	elif method == "eigenvector":#执行此句
		try:
			adj = to_scipy_sparse_matrix(graph.edge_index).tocsr()
			k = min(emb_method.dimensionality, graph.num_nodes - 2) #can only find N - 2 eigenvectors
			eigvals, eigvecs = sp_linalg.eigsh(adj.asfptype(), k = k)
			while eigvecs.shape[1] < emb_method.dimensionality:
				eigvecs = np.concatenate((eigvecs, eigvecs[:,-1].reshape((eigvecs.shape[0], 1))), axis = 1)
			eigvals = eigvals[:emb_method.dimensionality]
			eigvecs = eigvecs[:,:emb_method.dimensionality] 
		except Exception as e:
			print(e)
			#print(to_scipy_sparse_matrix(graph.edge_index).todense())
			eigvals, eigvecs = np.linalg.eig(to_scipy_sparse_matrix(graph.edge_index).todense())
			#append smallest eigenvector repeatedly if there are fewer eienvalues than embedding dimension
			while eigvecs.shape[1] < emb_method.dimensionality:
				eigvals = np.concatenate((eigvals, np.asarray([eigvals[-1]])))
				eigvecs = np.concatenate((eigvecs, eigvecs[:,-1].reshape((eigvecs.shape[0], 1))), axis = 1)
			eigvecs = eigvecs[:,np.argsort(-1*np.abs(eigvals))] #to match MATLAB
			eigvals = eigvals[:emb_method.dimensionality]
			eigvecs = eigvecs[:,:emb_method.dimensionality] 
		embeddings = np.abs(eigvecs)
	elif method == "rpf":
		embeddings = rpf(graph.adj, walk_length = emb_method.dimensionality)
	else:
		raise ValueError("Method %s not implemented yet" % method)

	#normalize, for graph similarity
	if emb_method.normalize:#特征分解不执行此句,xnetmf执行此步骤
		norms = np.linalg.norm(embeddings, axis = 1).reshape((embeddings.shape[0],1))
		norms[norms == 0] = 1 
		embeddings = embeddings / norms

	if emb_method.abs_val:#特征分解不执行此句，xnetmf执行此步骤
		embeddings = np.abs(embeddings)
	#print(embeddings)
	#print(emb_method.normalize)
	#print(emb_method.abs_val)
	return embeddings

def get_wl_labels(graphs, args, label_dict_by_iter = None):
	n_iter = args.wliter#初始值为2
	if label_dict_by_iter is None: #create empty list of dicts
		label_dict_by_iter = list()#存储两个字典
		for i in range(n_iter):
			label_dict_by_iter.append(dict())
	#Initialize labels to be the node labels
	before_wl_init = time.time()
	wl_labels =  [[] for i in range(n_iter + 1)]#第一个存储初始
	for j in range(len(graphs)):
		#if graphs[j].node_labels is None: 
		adj = to_scipy_sparse_matrix(graphs[j].edge_index).tocsr()
		wl_labels[0].append(np.ones(adj.shape[0]))#分配初始标签
	#print(wl_labels)
	print("WL label expansion time to initialize (iteration 0): ", (time.time() - before_wl_init))

	#Calculate new labels for WL
	for i in range(1, n_iter + 1): #One iteration of WL,执行2次
		before_wl_iter = time.time()
		label_num = 0 #each distinct combination of neighbors' labels will be assigned a new label, starting from 0 at each iteration
		for j in range(len(graphs)): #for each graph
			graph = graphs[j]
			wl_labels[i].append(list())
			adj = to_scipy_sparse_matrix(graphs[j].edge_index).tocsr()
			for k in range(adj.shape[0]): #for each node
				neighbors = adj[k].nonzero()[1] #get its neighbors
				neighbor_labels = wl_labels[i - 1][j][neighbors] #get their labels at previous iteration
				#prepend a node's own label, but sort neighbors' labels so that order doesn't matter
				neighbor_labels = np.insert(np.sort(neighbor_labels), 0, wl_labels[i - 1][j][k]) 

				#map these to a unique, order-independent string
				#this is a "label" for the node that is a multiset of its neighbors' labels
				#multiset_label = str(neighbor_labels)
				multiset_label = ''.join(map(str,neighbor_labels))

				#haven't encountered this label at this iteration
				if multiset_label not in label_dict_by_iter[i - 1]:
					#assign this a new numerical label that we haven't used at this iteration
					label_dict_by_iter[i - 1][multiset_label] = ("%d-%d") % (i, label_num) #new labeling number but also iteration number (so that we have all unique labels across iters)
					label_num += 1
				#For this iteration, assign the node a new WL label based on its neighbors' labels
				wl_labels[i][j].append(label_dict_by_iter[i - 1][multiset_label])#第j个子图
			wl_labels[i][j] = np.asarray(wl_labels[i][j])

		print("WL label expansion time at iteration %d: " % i, (time.time() - before_wl_iter))
	
	return wl_labels, label_dict_by_iter

#Input: length-N list of dict {str : int} of {d-dim coordinates of a cell : count}
#Output: N x D feature matrix, where D is the dimensionality of the feature mapping (# unique strings across all N items)
#NOTE: this is just for a single attribute 
#NOTE 2: will have to create training and test feature mappings together. 
#This seems to be OK, following (Rahimi an Recht, 2007).  
#The alternative is to create very long sparse vectors consisting of all (cell_width)^d possible cells
def combineddim_histogram_feature_map(combineddim_histograms):
	N = len(combineddim_histograms)
	#Get list of all unique non-empty cells
	combined_coordinates = set()
	for i in range(N):
		cells_hist = combineddim_histograms[i].keys()
		for cell in cells_hist:
			if cell not in combined_coordinates:
				combined_coordinates.add(cell)		

	#dimensionality of feature map
	D = len(combined_coordinates)#这个长度是变化的，也就是每一个子图统计出来的个数是在变化的，节点标签 节点嵌入作为整个数据集的身份标识，即节点标签 节点嵌入决定了整个图嵌入的
	#print(D)
	#map each cell (coordinates) to a dimension in the feature map
	coordinate_to_featuredim = dict(zip(list(combined_coordinates), range(D)))#构成一个字典
	#print(coordinate_to_featuredim)
	rows = list()
	cols = list()
	data = list()

	#go through all our graphs' histograms...
	for i in range(N):
		for coord in combineddim_histograms[i].keys(): #for each cell，对与单个图所拥有的节点标签 节点嵌入不同个数
			d = coordinate_to_featuredim[coord] #get the dimension in the feature map
			rows.append(i)
			cols.append(d)
			data.append(combineddim_histograms[i][coord])
	combined_feature_map = sparse.csr_matrix((data, (rows, cols)), shape = (N,D))
	return combined_feature_map

#Input: length-N list of n_i x d dimensional node embeddings for graph i
#Possibly: length-N list of n_i dimensional lists of labels for nodes in graph i
#Possibly: float or d-dimensional numpy array cell width
#Output: N x D feature matrix
def compute_feature_maps(embs,args,labels = None,cell_width = None):
	#Get coordinates for each embeddings
	if cell_width is None: #不执行此句
		#Get independent values along each dimension sampled from a gamma distribution with specified parameter
		cell_width = np.random.gamma(shape = 2, scale = 1.0/args.gammapitch, size = args.dimensionality)
		print("mean cell width %.3f" % (np.mean(cell_width)))#, cell_width
	#rgs.dimensionality=6
	offset = (np.random.uniform(size = args.dimensionality)*cell_width).tolist() #for each dimension, uniform between 0 and that dimension's cell width,长度为6
	#print(offset)
	#Turn embeddings into histograms (with or without labels)
	if labels is not None:#执行此句，有标签
		hists = get_labeled_embed_histograms(embs, args, cell_widths = cell_width, labels = labels, offsets = offset)#里面是列表字典，共有188个个字典表示188个子图
	else:
		hists = [get_histogram(emb, args, cell_widths = cell_width, offsets = offset) for emb in embs]

	#Turn histograms into feature maps
	#print(hists[1])#defaultdict(<class 'int'>, {'1.0 100110': 2, '1.0 200120': 1, '1.0 100030': 1, '1.0 100130': 1, '1.0 100120': 3, '1.0 200110': 1, '1.0 200030': 1, '1.0 100140': 1, '1.0 000120': 2})
	#print(len(hists))
	feature_maps = combineddim_histogram_feature_map(hists) #no longer just for combined dim#映射成188个子图的嵌入
	return feature_maps

#Compute feature maps (flattening the pyramid match kernel)
#Input: length-N list of n_i x d dimensional node embeddings for graph i
#Output: N x DL feature matrix, where L is the number of levels (args.numlevels)
def pyramid_feature_maps(embs, args, labels = None):
	feature_maps = None#args.numlevels=1#，这个参数调小可以
	for level in range(args.numlevels + 1): #Nikolentzos et. al count levels from 0 to L inclusive，执行2次。这个参数
		cell_width = np.random.gamma(shape = 2, scale = 1.0/2**(level+1), size = args.dimensionality)
		#print("Cell width level %d (mean cell width %.3f)" % (level, np.mean(cell_width)))#, cell_width
		discount_factor = np.sqrt(1.0/2**(args.numlevels - level)) #discount feature maps corresponding to looser grids, as those matches are considered less important in PM
		features_level = discount_factor * compute_feature_maps(embs, args, labels = labels, cell_width = cell_width)#这一句已经生成了188个子图的嵌入
		#print(features_level.shape)#188×18
		if feature_maps is None: 
			feature_maps = features_level
		else:#第一次循环不执行此句
			if sparse.issparse(feature_maps):
				feature_maps = sparse.hstack((feature_maps, features_level), format = "csr")
			else:
				feature_maps = np.hstack((feature_maps, features_level))
	return feature_maps

#Full RGM-P or L with up to P normalized randomized feature maps
#Input: length-N list of n_i x d dimensional node embeddings for graph i
#Output: N x DL feature matrix, where P is the number of trials (args.numrmaps)
def rgm(embs, args, labels = None):
	#print(labels)#labels[1]的格式array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])，表示第一个子图的节点标签
	#print(args.dimensionality)
	return pyramid_feature_maps(embs, args, labels = labels)