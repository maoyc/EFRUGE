import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnEdge Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset',default= ['MUTAG'])#采用的数据为：MUTAG、ENZYMES、IMDB-BINARY、ogbg-molhiv['MUTAG','IMDB-BINARY','ENZYMES','ogbg-molhiv']
    #parser.add_argument('--type', help='[edge_gin,gin]',default= 'no_graph')#no_graph,deepwalk,gin,
    parser.add_argument('--prior', dest='prior', action='store_const', 
            const=True, default=True)#原始为false
    parser.add_argument('--model_name', help='[edge_gin,gin]',default= 'node2vec')#xnetmf,no_graph,deepwalk,gin,node2vec,graph2vec,dgcnn,gcn,diffpool,rgm,efruge_end_to_end
    parser.add_argument('--lr', dest='lr', type=float,default=0.1,
            help='Learning rate.')
    parser.add_argument('--learn_eps', dest='learn_eps', type=float,default=0.0,
            help='€-value')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')
    parser.add_argument('--output_dim', dest='output_dim', type=int, default=32,
            help='')
    parser.add_argument('--graph_pooling_type', dest='graph_pooling_type', type=str, default='mean',
            help='')#sum,mean,max
    parser.add_argument('--neighbor_pooling_type', dest='neighbor_pooling_type', type=str, default='sum',
            help='')#sum,mean,max
    parser.add_argument('--measure', dest='measure', type=str, default='JSD',
            help='GAN,JSD,X2,KL,RKL,DV,H2,W1')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5,
            help='')#模型训练的次数,默认500
    parser.add_argument('--train', dest='train', type=bool, default=False,
            help='')#是否模型训练
    parser.add_argument('--vis', dest='vis', type=bool, default=True,
            help='')#是否嵌入可视化
    parser.add_argument("--input-path",
                        nargs="?",
                        default="./dataset/",
	                help="Input folder with jsons.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="./features/nci1.csv",
	                help="Embeddings path.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=128,
	                help="Number of dimensions. Default is 128.")

    parser.add_argument("--workers",
                        type=int,
                        default=1,
	                help="Number of workers. Default is 4.")

    parser.add_argument("--epochs_graph2vec",
                        type=int,
                        default=10,
	                help="Number of epochs. Default is 10.")

    parser.add_argument("--min-count", type=int,default=5,help="Minimal structural feature count. Default is 5.")                   
    parser.add_argument("--wl-iterations",type=int,default=2,help="Number of Weisfeiler-Lehman iterations. Default is 2.")                       
    parser.add_argument("--learning-rate",type=float,default=0.025,help="Initial learning rate. Default is 0.025.")                
    parser.add_argument("--down-sampling",type=float,default=0.0001,help="Down sampling rate of features. Default is 0.0001.")                                  
    parser.add_argument('--emb', nargs='?', default='xnetmf', help='Embedding method (xnetmf, eigenvector, struc2vec)')
    parser.add_argument('--dimensionality', nargs='?', type = int, default=100, help='dimensionality of embeddings')
    parser.add_argument('--wliter', nargs='?', type = int, default=2, help="Number of iterations of Weisfeiler-Lehman kernel. Default is 0 (no WL kernel)")
    parser.add_argument('--numlevels', nargs='?', type = int, default=1, help="Number of levels for pyramid match kernel")
    return parser.parse_args()