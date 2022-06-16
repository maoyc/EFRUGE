"""Graph2Vec module."""

import os
import json
import glob
import hashlib
import pandas as pd
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
from torch_geometric.utils import to_networkx
#from param_parser import parameter_parser
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations):
        """
        Initialization method which also executes feature extraction.
        :param graph: The Nx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]#节点的特征全是字符，类似于构造一个文本。
        #print(len(self.extracted_features))
        self.do_recursions()
        

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)#将特征用'_'连起来。
            #print(features.encode())
            hash_object = hashlib.md5(features.encode())
            hashing = hash_object.hexdigest()
            new_features[node] = hashing#hashing是一堆字符串
            #print(hashing)
        self.extracted_features = self.extracted_features + list(new_features.values())
        #print(len(self.extracted_features) ),self.extracted_features的长度是原始特征的三倍，
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()

def path2name(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def dataset_reader(path):
    """
    Function to read the graph and features from a json file.
    :param path: The path to the graph json.
    :return graph: The graph object.
    :return features: Features hash table.
    :return name: Name of the graph.
    """
    name = path2name(path)
    #print(path)
    #print(name)
    data = json.load(open(path))
    graph = nx.from_edgelist(data["edges"])

    if "features" in data.keys():
        features = data["features"]#每一个节点的特征是一个单一的值,返回的是一个字典
        features = {int(k): v for k, v in features.items()}
    else:
        features = nx.degree(graph)
        features = {int(k): v for k, v in features}
       
    return graph, features, name

def feature_extractor(path, name,rounds):
    """
    Function to extract WL features from a graph.
    :param path: The data to the dataset pyg.
    :param rounds: Number of WL iterations.
    :return doc: Document collection object.
    """
    graph = to_networkx(path,to_undirected=True)
    #print(torch.argmax(path.x,dim=1))
    features = [str(i.item()) for i in torch.argmax(path.x,dim=1)]#首先节点特征要存在
    features = {int(k):v for k,v in enumerate(features)}
    #graph, features, name = dataset_reader(path)
    #print(features)#features 是字典形式，key是整数节点，value是字符
    machine = WeisfeilerLehmanMachine(graph, features, rounds)
    name = str(name)
    doc = TaggedDocument(words=machine.extracted_features, tags=["g_" + name])#也就是说每一个子图的特征是由节点特征迭代三次以后生成的一个特征，类似于构造一个文章，
    #print(doc)
    return doc#是一个元组，第一个是一个文档，第二个是名字，元组的每一个元素是列表

def save_embedding(output_path, model, files, dimensions):
    """
    Function to save the embedding.
    :param output_path: Path to the embedding csv.
    :param model: The embedding model object.
    :param files: The list of files.
    :param dimensions: The embedding dimension parameter.
    """
    out = []
    for f in files:
        identifier = path2name(f)
        out.append([identifier] + list(model.docvecs["g_"+identifier]))#out里面的元素第一个是字符，剩下的是数值，相当于是一个图的嵌入。
        #print(out[0])
        #break
    column_names = ["type"]+["x_"+str(dim) for dim in range(dimensions)]
    #print(column_names)
    out = pd.DataFrame(out, columns=column_names)
    out = out.sort_values(["type"])
    out.to_csv(output_path, index=None)

def main_graph2vec(args):
    """
    Main function to read the graph list, extract features.
    Learn the embedding and save it.
    :param args: Object with the arguments.
    """
    graphs = glob.glob(os.path.join(args.input_path, "*.json"))
    print("\nFeature extraction started.\n")
    #print(graphs)#默认args.wl_iterations=2
    document_collections = Parallel(n_jobs=args.workers)(delayed(feature_extractor)(g, args.wl_iterations) for g in tqdm(graphs))
    #print(document_collections)
    print("\nOptimization started.\n")

    model = Doc2Vec(document_collections,
                    vector_size=args.dimensions,
                    window=0,
                    min_count=args.min_count,
                    dm=0,
                    sample=args.down_sampling,
                    workers=args.workers,
                    epochs=args.epochs,
                    alpha=args.learning_rate)
    #print(model)
    #print(model.docvecs['g_1'])
    save_embedding(args.output_path, model, graphs, args.dimensions)

if __name__ == "__main__":
    args = parameter_parser()
    args.input_path = 'D:/Users/DELL/source/repos/PythonApplication4/graph2vec-master/dataset'
    print(args)
    main_graph2vec(args)

