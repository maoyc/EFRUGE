from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

def draw_plot(datadir, DS, embeddings, fname, max_nodes=None):
    return
    import seaborn as sns
    graphs = read_graphfile(datadir, DS, max_nodes=max_nodes)
    labels = [graph.graph['label'] for graph in graphs]

    labels = preprocessing.LabelEncoder().fit_transform(labels)
    x, y = np.array(embeddings), np.array(labels)
    print('fitting TSNE ...')
    x = TSNE(n_components=2).fit_transform(x)

    plt.close()
    df = pd.DataFrame(columns=['x0', 'x1', 'Y'])

    df['x0'], df['x1'], df['Y'] = x[:,0], x[:,1], y
    sns.pairplot(x_vars=['x0'], y_vars=['x1'], data=df, hue="Y", size=5)
    plt.legend()
    plt.savefig(fname)

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):#这种参数初始化的操作有没有必要
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

def logistic_classify(x, y):
    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()#交叉熵损失函数
    hid_units = x.shape[1]

    accs = []
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=None)#默认为10
    for train_index, test_index in kf.split(x, y):#划分训练集和测试集，就是所谓交叉验证的次数，返回的是索引，
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()


        log = LogReg(hid_units, nb_classes)#回归模型，
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()
        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())
    return np.mean(accs)

def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)#参数评估器，用于寻找最优的参数，自动调参
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        #print(classifier.predict(x_test))
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)

def beiyesi_classify(x,y):
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #print(x_train)
        #print(y_train)
        #print(x_train.shape)
        #print(y_train.shape)
        classifier = GaussianNB()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    return np.mean(accuracies)
def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    ret = np.mean(accuracies)
    return ret


def evaluate_embedding(embeddings, labels, search=True):
    #print(labels.shape)
    if type(labels) is np.ndarray:
        labels = preprocessing.LabelEncoder().fit_transform(labels)
        x,y = embeddings.astype(np.float32),labels
    else:
        labels = preprocessing.LabelEncoder().fit_transform(labels.cpu().numpy())#labels没有维度，并将标签转换为整数。将标签值统一转换成range(标签个数-1)范围内。
        x, y = np.array(embeddings.cpu()), np.array(labels)
    #df = pd.DataFrame(x)
    #print(df[df.isnull().T.any()])
    #print(df.head(5))
    #print(x.dtype)
    #print(y.dtype)


    logreg_accuracies = [logistic_classify(x, y) for _ in range(10)]#计算的次数，每次都交叉验证
    #print(logreg_accuracies)
    print('LogReg', np.mean(logreg_accuracies))#线性回归

    svc_accuracies = [svc_classify(x,y, search) for _ in range(10)]
    #print(svc_accuracies)
    print('svc', np.mean(svc_accuracies))#支持向量机分类模型

    beiyes_accuracies = [beiyesi_classify(x,y) for _ in range(10)]
    #print(beiyes_accuracies)
    print('beiyes',np.mean(beiyes_accuracies))

    randomforest_accuracies = [randomforest_classify(x, y, search) for _ in range(10)]
    #print(randomforest_accuracies)
    print('randomforest', np.mean(randomforest_accuracies))

    #return np.mean(logreg_accuracies), np.mean(svc_accuracies), np.mean(linearsvc_accuracies), np.mean(randomforest_accuracies)
    return logreg_accuracies,svc_accuracies,beiyes_accuracies,randomforest_accuracies

if __name__ == '__main__':
    #evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
    labels = torch.tensor([[1],[2],[4],[6],[1]])
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    print(labels)

