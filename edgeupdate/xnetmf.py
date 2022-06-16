import numpy as np, scipy as sp, networkx as nx
import math, time, os, sys
from torch_geometric.utils import k_hop_subgraph,to_networkx
import torch
import matplotlib.pyplot as pltpython 
import torch_geometric.utils

#Input: graph, RepMethod
#Output: dictionary of dictionaries: for each node, dictionary containing {node : {layer_num : [list of neighbors]}}
#        dictionary {node ID: degree}
def get_khop_neighbors(graph, emb_method):
	kneighbors_dict = {}
	#only 0-hop neighbor of a node is itself
	#neighbors of a node have nonzero connections to it in adj matrix
	for node in range(graph.num_nodes):

		one_hop,_,_,_ = k_hop_subgraph(node,1,graph.edge_index)
		two_hop,_,_,_ = k_hop_subgraph(node,2,graph.edge_index)

		one_hop = one_hop.cpu().numpy().tolist()
		two_hop = two_hop.cpu().numpy().tolist()

		kneighbors_dict[node] = {0: set([node]), 1: set(one_hop)-set([node]),2:set(two_hop)-set(one_hop)}

	return kneighbors_dict


#Turn lists of neighbors into a degree sequence
#Input: graph, RepMethod, node's neighbors at a given layer, the node
#Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
def get_degree_sequence(graph, emb_method, kneighbors, current_node,num_features,G):
	#if emb_method.num_buckets is not None:
	#	degree_counts = [0] * int(math.log(graph.max_features["degree"], emb_method.num_buckets) + 1)
	#else:
	degree_counts = [0] * num_features
	#print(graph.edge_index)
	#max_degree = torch_geometric.utils.degree(graph.edge_index[0],graph.num_nodes).max()#是按照无向图进行统计的
	#print(max_degree)
	#For each node in k-hop neighbors, count its degree
	#print(kneighbors)
	for kn in kneighbors:
		weight = 1 #unweighted graphs supported here
		degree = G.degree(kn)
		if degree >= len(degree_counts): #larger degree than these buckets were created for
			degree_counts[-1] += weight #add to last bucket, which captures all the largest degrees
		else:
			degree_counts[degree] += weight
	#print(degree_counts)
	return degree_counts

#Get structural features for nodes in a graph based on degree sequences of neighbors
#Input: graph, RepMethod
#Output: nxD feature matrix
def get_features(graph, emb_method):
	before_khop = time.time()
	#Get k-hop neighbors of all nodes
	khop_neighbors_nobfs = get_khop_neighbors(graph, emb_method)
	#print(khop_neighbors_nobfs)
	#graph.khop_neighbors = khop_neighbors_nobfs

	num_nodes = graph.num_nodes
	G = to_networkx(graph,to_undirected=True)
	num_features = sorted(G.degree(),key=lambda i:i[1])[-1][1]+1
	#print(num_features)
	#nx.draw(G,with_labels=True)
	#plt.show()

	feature_matrix = np.zeros((num_nodes, num_features))
	#print(graph.khop_neighbors[0])#节点0对应的h跳层邻居节点{0: {0}, 1: {1, 5}, 2: {2, 4}}
	before_degseqs = time.time()
	for n in range(num_nodes):
		for layer in khop_neighbors_nobfs[n].keys(): #construct feature matrix one layer at a time
			if len(khop_neighbors_nobfs[n][layer]) > 0:
				#degree sequence of node n at layer "layer"
				deg_seq = get_degree_sequence(graph, emb_method, khop_neighbors_nobfs[n][layer],n,num_features,G)#n为节点的编号，返回的是一个度序列，就是统计每个度下节点的个数
				#add degree info from this degree sequence, weighted depending on layer and discount factor alpha
				feature_matrix[n] += [(emb_method.alpha**layer) * x for x in deg_seq]
	after_degseqs = time.time() 

	return feature_matrix

#Input: two vectors of the same length
#Optional: tuple of (same length) vectors of node attributes for corresponding nodes
#Output: number between 0 and 1 representing their similarity
def compute_similarity(graph, emb_method, vec1, vec2, node_indices = None):
	dist = emb_method.gammastruc * np.linalg.norm(vec1 - vec2) #compare distances between structural identities,返回范数
	#print(graph.node_attributes)
	#print(dist)
	#print(node_indices[0])
	#print(node_indices[1])
	#print(graph.x[node_indices[0]])
	#print(graph.x[node_indices[1]])
	#print(graph.x[node_indices[0]] != graph.x[node_indices[1]])
	attr_dist = torch.sum(graph.x[node_indices[0]] != graph.x[node_indices[1]])
	dist += emb_method.gammaattr * attr_dist
	#print(dist)
	#print(np.exp(-dist))
	
	return np.exp(-dist.item()) #convert distances (weighted by coefficients on structure and attributes) to similarities

#Sample landmark nodes (to compute all pairwise similarities to in Nystrom approx)
#Input: graph (just need graph size here), RepMethod (just need dimensionality here)
#Output: np array of node IDs
def get_sample_nodes(graph, emb_method):
	#Sample uniformly at random
	sample = np.random.permutation(np.arange(graph.num_nodes))[:emb_method.dimensionality]
	#print(sample)
	return sample#返回节点的编号

#xNetMF pipeline
def get_representations(graph, emb_method):#传进来已经是一个批次的大图。
	#Node identity extraction
	feature_matrix = get_features(graph, emb_method)#节点个数×节点个数，所以这一步是关键
	#print(feature_matrix.shape)
	#Efficient similarity-based representation
	#Get landmark nodes, unless using previously computed ones
	#if not (emb_method.method == "xnetmf" and emb_method.landmark_features is not None and emb_method.use_landmarks):
	#print(emb_method.landmark_features)
	#print(emb_method.use_landmarks)
	if emb_method.landmark_features is None or not emb_method.use_landmarks:#执行此句
		print("Getting landmark features...")

		landmarks = get_sample_nodes(graph, emb_method)#采集的节点个数就是嵌入的节点维度。
		emb_method.landmark_features = feature_matrix[landmarks]
		emb_method.landmark_indices = landmarks
		#print(emb_method.landmark_features)#一个二维数组，对应节点的特征
	#Explicitly compute similarities of all nodes to these landmarks
	before_computesim = time.time()
	C = np.zeros((graph.num_nodes,emb_method.dimensionality))
	for node_index in range(graph.num_nodes): #for each of N nodes
		for landmark_index in range(emb_method.dimensionality): #for each of p landmarks，采样的节点个数等于其维度，所以landmark_node_features是存在的
			#select the p-th landmark
			if emb_method.landmark_features is not None: #landmarks are hard coded as actual features of landmark nodes，执行此句
				landmark_node_features = emb_method.landmark_features[landmark_index]
			else:
				landmark_node_features = feature_matrix[landmarks[landmark_index]] #landmarks are indices of landmark nodes
			C[node_index,landmark_index] = compute_similarity(graph,emb_method,feature_matrix[node_index],landmark_node_features,(node_index, landmark_index))   														
	#print(C)
	before_computerep = time.time()

	#Compute Nystrom-based node embeddings
	#print(emb_method.l2l_decomp)none
	#print(emb_method.use_landmarks)false
	if emb_method.l2l_decomp is None or not emb_method.use_landmarks:#执行此句
		#Compute and factorize SVD of pseudoinverse of landmark-to-landmark similarity matrix, unless we have one already
		W_pinv = np.linalg.pinv(C[landmarks])
		U,X,V = np.linalg.svd(W_pinv)
		Wfac = np.dot(U, np.diag(np.sqrt(X)))
		emb_method.l2l_decomp = Wfac

	#print(emb_method.l2l_decomp.shape)#嵌入维度×嵌入维度
	reprsn = np.dot(C, emb_method.l2l_decomp)#生成了节点个数×嵌入维度
	after_computerep = time.time()

	#Post-processing step to normalize embeddings (true by default, for use with REGAL)
	if emb_method.normalize:
		reprsn = reprsn / np.linalg.norm(reprsn, axis = 1).reshape((reprsn.shape[0],1))
	#print(reprsn.shape)
	return reprsn#返回的是一个完整大图的嵌入

if __name__ == "__main__":
	if len(sys.argv) < 2:
		#####PUT IN YOUR GRAPH AS AN EDGELIST HERE (or pass as cmd line argument)#####  
		#(see networkx read_edgelist() method...if networkx can read your file as an edgelist you're good!)
		graph_file = "data/arenas_combined_edges.txt"
	else:
		graph_file = sys.argv[1]
	nx_graph = nx.read_edgelist(graph_file, nodetype = int, comments="%")
	adj_matrix = nx.adjacency_matrix(nx_graph).todense()
	
	graph = Graph(adj_matrix)
	emb_method = RepMethod(max_layer = 2) #Learn representations with xNetMF.  Can adjust parameters (e.g. as in REGAL)
	representations = get_representations(graph, emb_method)
	print(representations.shape)










