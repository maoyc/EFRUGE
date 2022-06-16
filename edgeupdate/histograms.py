#All code for constructing embedding histograms, used in PM kernel and RGM feature maps
import numpy as np 
from collections import defaultdict
from itertools import product

'''*********************Creat discretized histograms for real-valued embeddings*********************'''

#Get integer coordinates for each dimension of each embedding
#Input: emb (n x d numpy array)
#cell widths: float (for same width along all dimensions), or d-dimensional numpy array specifying width along each dim
#offsets: int or float float (for same offset along all dimensions), or d-dimensional numpy array specifying offset along each dim
#Output: coord (n x d array of ints): coord[i,j] says into which cell node i falls along dimension j
def get_coordinates(emb, cell_widths, offsets = 0):
	try:
		n, d = emb.shape#返回节点个数和节点嵌入长度
	except:
		raise ValueError(emb.shape)

	#Elementwise matrix operations if cell widths and offsets are not scalars
	if type(cell_widths) is list or type(cell_widths).__module__ == np.__name__:
		cell_widths = np.repeat(np.asarray(cell_widths).reshape((len(cell_widths), 1)), n, axis = 1).T
	if type(offsets) is list or type(offsets).__module__ == np.__name__:
		offsets = np.repeat(np.asarray(offsets).reshape((len(offsets), 1)), n, axis = 1).T

	coord = np.floor((emb + cell_widths - offsets)/cell_widths).astype(int)#返回节点个数×嵌入维度6，但是每一个维度的位置变成了整数
	return coord


#Converts coordinates into histograms
#Each of n embeddings is mapped to one cell based on the coordinates of all d of its dimensions
#Input: n x d matrix of coordinates for each embedding
#Output: dict {str : int} of {d-dim coordinates of a cell : count}
def get_combineddim_histograms(coord, label = None, dict_grid = None):
	n, d = coord.shape
	if dict_grid is None: dict_grid = defaultdict(int)
	for i in range(n):#表示该标签在该图中的个数，遍历每一个节点
		node_coordinate = ''.join(map(str,coord[i]))#将该节点的嵌入拼接起来
		if label is not None:
			node_coordinate = ("%s %s") % (str(label), node_coordinate) #prepend this label to create a new histogram bin based on these coordinates but for this label
		dict_grid[node_coordinate] += 1
	return dict_grid#返回同一个标签下，节点嵌入也相同的个数


#Wrapper function to make histograms of either format
#Input: emb (n x d numpy array)
#cell widths: float (for same width along all dimensions), or d-dimensional numpy array specifying width along each dim
#offsets: int or float float (for same offset along all dimensions), or d-dimensional numpy array specifying offset along each dim
#indiv_dim: boolean flag whether to perform binning by node or by dimension
#label: (str or int) the label of these embeddings (if desired)
#dict_grid: dict {str : int} of {d-dim coordinates of a cell : count} histogram to add to, if desired
#Output: dict {int : dict{int: int}} of {dimension : {cell number : count}} if indivdim 
#		dict {str : int} of {d-dim coordinates of a cell : count} if combineddim
def get_histogram(emb, args, cell_widths, offsets = 0, label = None, dict_grid = None):
	#All embeddings have no label / the same label
	coord = get_coordinates(emb, cell_widths, offsets) #get embedding coordinates based on grid，某个标签在某个图中的嵌入n×d。n表示该标签在该图中的节点个数。
	#compute desired histogram
	return get_combineddim_histograms(coord, label = label, dict_grid = dict_grid)#返回的事一个字典{'节点标签':个数}


#Wrapper function to make histograms of either format
#Input: length-N list of embs (n x d real-valued numpy array)
#cell widths: float (for same width along all dimensions), or d-dimensional numpy array specifying width along each dim
#labels: length-N list of n_i-dimensional node embeddings for graph i
#offsets: int or float float (for same offset along all dimensions), or d-dimensional numpy array specifying offset along each dim
#Output: list of dict {int : dict{int: int}} of {dimension : {cell number : count}} if indivdim 
#		list of dict {str : int} of {d-dim coordinates of a cell : count} if combineddim
def get_labeled_embed_histograms(embs, args, cell_widths, labels = None, offsets = 0):
	embs_by_label = partition_embs_labels(embs, labels) #{int/str : {int : numpy array}} label: {graph id: embeddings with that id}
	emb_hists = [defaultdict(int) for i in range(len(embs))] #list of histograms for each graph
	for label in embs_by_label: #for all labels
		for graph_id in embs_by_label[label]: #for all graphs that have embeddings with this label
			#compute its histogram for its embeddings with this label and add them into its overall histogram
			emb_hists[graph_id] = get_histogram(embs_by_label[label][graph_id], 
												args,
												cell_widths = cell_widths, 
												offsets = offsets, 
												label = label, 
												dict_grid = emb_hists[graph_id])#相当于向字典里追加
	return emb_hists



'''*********************Split up embeddings for labeled graphs*********************'''

#For a graph, return all unique labels mapped to embedding matrix of all nodes with that label
#Input: n_i x d dimensional np array node embeddings for graph i
#       n_i-dimensional np array node labels for graph i
#Output: {int/str : np array} dict of labels matched to embeddings with that label
def split_graph_embs_by_label(emb, labels):#一个图的嵌入，一个图的标签
	embs_by_label = {}
	for label in labels: #for each label
		if label not in embs_by_label: #if this is a new label, otherwise we've already taken care of it
			embs_by_label[label] = emb[labels == label] #select out embeddings of nodes with that label
	return embs_by_label


#For several sets of graphs' embeddings grouped by labels, map each unique label to 
#  (all graphs ID with nodes that have that label mapped to their embeddings with that label)
#Input: length-N list of {int : np array} dict of labels matched to embeddings with that label
#Output: {int : {int : numpy array}} dict of labels matched to a dict of graph IDs to embeddings in that graph with that label
def group_embs_by_label(labeled_embs):
	embs_by_label = defaultdict(dict)
	for i in range(len(labeled_embs)): #for each graph
		graph_labeled_embs = labeled_embs[i]
		for label in graph_labeled_embs: #for each label in that graph
			#in the final collection under that label, add that graph ID mapped to its embeddings with that label
			embs_by_label[label][i] = graph_labeled_embs[label]
	return embs_by_label

#Split and regroup embs by label
#Input: lst where the i-th entry is n_i x d dimensional np array node embeddings for graph i
#       list where the i-th entry is the n_i-dimensional np array node embeddings for graph i
#Output: {int : {int : numpy array}} dict of labels matched to a dict of graph IDs to embeddings in that graph with that label
def partition_embs_labels(embs, labels):
	split_embs = []
	for i in range(len(embs)):
		split_embs.append(split_graph_embs_by_label(embs[i], labels[i]))
	embs_by_label = group_embs_by_label(split_embs)
	return embs_by_label
