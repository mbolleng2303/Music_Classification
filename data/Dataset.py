import csv
import os
import pickle
from tqdm import tqdm
import dgl as dgl
import numpy as np
import torch as torch
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
random.seed(42)


class Data2Features (torch.utils.data.Dataset):
    def __init__(self, name='img', data_dir="../data", data_feat="random"):
        self.name = name
        self.data_dir = data_dir
        self.data_feat = data_feat
        self.features_lists = []
        self.labels_lists = []
        self._prepare()

    def _prepare(self):
        t0 = time.time()
        print("[I] Preparing features...")
        if self.data_feat != 'random':
            print('creating new dataset')
            if self.data_feat == 'mfcc':
                self.features = np.reshape(np.load(self.data_dir +'/'+'mfcc_tagtraum_clean.npy'), (-1, 1, 300, 12))
                self.nbr_sample = len(self.features[:, 0, 0, 0])
                self.nbr_feature = len(self.features[0, :, 0, 0])
                self.size = (self.features.shape[2], self.features.shape[3])
                self.labels = np.load(self.data_dir +'/'+'label.npy')
                self.nbr_classes = len(np.unique(self.labels))
                assert self.nbr_sample == len(self.labels), 'Problem ! :('
            elif self.data_feat == 'chroma':
                self.features = np.reshape(np.load(self.data_dir +'/'+'chroma_tagtraum_clean.npy'), (-1, 1, 300, 12))
                self.nbr_sample = len(self.features[:, 0, 0, 0])
                self.nbr_feature = len(self.features[0, :, 0, 0])
                self.size = (self.features.shape[2], self.features.shape[3])
                self.labels = np.load(self.data_dir +'/'+'label.npy')
                self.nbr_classes = len(np.unique(self.labels))
                assert self.nbr_sample == len(self.labels), 'Problem ! :('
            elif self.data_feat =='mfcc_chroma':
                self.mfcc = np.reshape(np.load(self.data_dir + '/' + 'mfcc_tagtraum_clean.npy'), (-1, 1, 300, 12))
                self.chroma = np.reshape(np.load(self.data_dir + '/' + 'chroma_tagtraum_clean.npy'),
                                         (-1, 1, 300, 12))

                self.features = np.concatenate((self.mfcc, self.chroma), axis=1)
                self.nbr_sample = len(self.features[:, 0, 0, 0])
                self.nbr_feature = len(self.features[0, :, 0, 0])
                self.size = (self.features.shape[2], self.features.shape[3])
                self.labels = np.load(self.data_dir + '/' + 'label.npy')
                self.nbr_classes = len(np.unique(self.labels))
                assert self.nbr_sample == len(self.labels), 'Problem ! :('


        # TODO : prepare de features in the expected modality
        else:
            self.features = np.reshape(
                np.loadtxt(self.data_dir + '/img.csv',
                           delimiter=",", dtype=float), (-1, 3, 128, 128))
            self.nbr_sample = len(self.features[:, 0, 0, 0])
            self.nbr_feature = len(self.features[0, :, 0, 0])
            self.size = (self.features.shape[2], self.features.shape[3])
            self.labels = np.reshape(
                np.loadtxt(self.data_dir + '/label.csv',
                           delimiter=",", dtype=float), (-1))
            self.nbr_classes = len(np.unique(self.labels))
            assert self.nbr_sample == len(self.labels), 'Problem ! :('
        temp = np.zeros(self.nbr_classes)
        for i in range(self.nbr_sample):
            temp[int(self.labels[i])] = 1
            self.labels_lists.append(temp)
            self.features_lists.append(self.features[i, :, :, :])
            temp = np.zeros(self.nbr_classes)
        print("[I] Finished preparation after {:.4f}s".format(time.time() - t0))

    def __len__(self):
        return self.nbr_sample

    def __getitem__(self, idx):
        return self.features_lists[idx], self.labels_lists[idx], Idx(idx)

def get_vertices(a):
    edges = []
    feat = []
    for i in range(a.shape[1]):
        for j in range(0, a.shape[0]):#i
            if a[i, j] != 0:
                edges.append((i, j))
                feat.append(a[i, j])
                # edges.append((j, i)) #for two dir
    return edges, feat

class Data2Graph2(torch.utils.data.Dataset):


    def __init__(self, name='img', data_dir="../data", data_feat="random"):
        self.name = name
        self.data_dir = data_dir
        self.data_feat = data_feat
        self.nbr_graphs = 30
        self.nbr_node = 100
        self.name = name
        if self.data_feat =='chroma':
            self.graph = np.load(self.data_dir+'/'+'graph_chroma.npy')
        else:
            self.graph = np.load(self.data_dir + '/' + 'graph_mfcc.npy')
        self.label = np.load(self.data_dir+'/'+'graph_label.npy')
        self.nbr_classes = 15
        self.edge = np.load(self.data_dir+'/'+'graph_edges.npy')
        self.graph_lists = []
        self.label_lists = []
        self.labels_lists = []
        self.features_lists = []
        self._prepare()

    def _prepare(self):
        t0 = time.time()
        print("[I] Preparing graph :)...")
        for i in tqdm(range (self.nbr_graphs)):
            g = dgl.DGLGraph()
            g.add_nodes(self.nbr_node)
            g.ndata['feat'] = torch.tensor(self.graph[:,i,:].T).long()
            edge = np.array(get_vertices(self.edge[i,:,:])[0])
            edge_feat = np.array(get_vertices(self.edge[i,:,:])[1])
            for src, dst in edge:
                g.add_edges(src.item(), dst.item())
            edge_feat_dim = 1
            edge_feat=np.array(edge_feat)
            g.edata['feat'] = torch.tensor((edge_feat)).long()
            g = dgl.transform.remove_self_loop(g)
            a = (self.label[0, i, :])
            res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            b= []
            for i in a:
                res[int(i)]=1
                b.append(res)
                res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.label_lists.append(torch.tensor(b))
            self.labels_lists.append(torch.tensor(b))
            self.graph_lists.append(g)
            self.features_lists.append(g)
        print("[I] Finished preparation after {:.4f}s".format(time.time() - t0))
    def __len__(self):
        return self.nbr_graphs

    def __getitem__(self, idx):
        return self.features_lists[idx], self.labels_lists[idx], Idx(idx)
class Data2Graph(torch.utils.data.Dataset):


    def __init__(self, name='img', data_dir="../data", data_feat="random"):
        self.name = name
        self.data_dir = data_dir
        self.data_feat = data_feat
        self.nbr_graphs = 3011

        self.name = name
        if self.data_feat == 'chroma':
            self.nbr_node = 12
            self.graph = np.load(self.data_dir+'/'+'graph_chroma_a.npy')
        elif self.data_feat =='mfcc':
            self.nbr_node = 12
            self.graph = np.load(self.data_dir + '/' + 'graph_mfcc_a.npy')
        elif self.data_feat =='mfcc_chroma':
            self.nbr_node = 24
            self.graph_mfcc = np.load(self.data_dir + '/' + 'graph_mfcc_a.npy')
            self.graph_chroma = np.load(self.data_dir + '/' + 'graph_mfcc_a.npy')
        self.label = np.load(self.data_dir+'/'+'graph_label_a.npy')
        self.nbr_classes = 15
        self.edge = np.load(self.data_dir+'/'+'graph_edges_a.npy')
        self.graph_lists = []
        self.label_lists = []
        self.labels_lists = []
        self.features_lists = []
        self._prepare()

    def _prepare(self):
        t0 = time.time()
        print("[I] Preparing graph :)...")
        for i in tqdm(range(self.nbr_graphs)):
            g = dgl.DGLGraph()
            if self.data_feat != 'mfcc_chroma':
                g.add_nodes(self.nbr_node)
                g.ndata['feat'] = torch.tensor(self.graph[:, :, i]).long()
                edge = np.array(get_vertices(self.edge[i,:,:])[0])
                edge_feat = np.array(get_vertices(self.edge[i,:,:])[1])
                for src, dst in edge:
                    g.add_edges(src.item(), dst.item())
                edge_feat=np.array(edge_feat)
                g.edata['feat'] = torch.tensor(edge_feat).long()
                #g = dgl.transform.remove_self_loop(g)
            else :
                g.add_nodes(self.nbr_node)
                g.ndata['feat'] = torch.tensor(np.concatenate((self.graph_mfcc[:, :, i], self.graph_chroma[:, :, i]), axis =0)).long()
                one = np.ones((12, 12))
                zero = np.zeros((12, 12))
                mat1 = np.concatenate((one,zero), axis=1)
                mat2 = np.concatenate((zero,one), axis=1)
                mat = np.concatenate((mat1,mat2), axis=0)
                edge = np.array(get_vertices(mat)[0])
                edge_feat = np.array(get_vertices(mat)[1])
                for src, dst in edge:
                    g.add_edges(src.item(), dst.item())
                edge_feat = np.array(edge_feat)
                g.edata['feat'] = torch.tensor(edge_feat).long()
            a = (self.label[0,i])
            res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            b= []

            res[int(a)]=1
            b.append(res)
            res = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.label_lists.append(torch.tensor(b))
            self.labels_lists.append(torch.tensor(b))
            self.graph_lists.append(g)
            self.features_lists.append(g)
        print("[I] Finished preparation after {:.4f}s".format(time.time() - t0))
    def __len__(self):
        return self.nbr_graphs

    def __getitem__(self, idx):
        return self.features_lists[idx], self.labels_lists[idx], Idx(idx)


class Idx:
    def __init__(self, idx):
        self.a = self
        self.index = idx


def get_all_split_idx(dataset, data_dir, k_splits):
        """
            - Split total number of features into 3 (train, val and test) in 3:1:1
            - Stratified split proportionate to original distribution of data with respect to classes
            - Using sklearn to perform the split and then save the indexes
            - Preparing 5 such combinations of indexes split to be used in Graph NNs
            - As with KFold, each of the 5 fold have unique test set.
        """
        root_idx_dir = data_dir + '/split/'
        if not os.path.exists(root_idx_dir):
            os.makedirs(root_idx_dir)
        all_idx = {}

        # If there are no idx files, do the split and store the files
        if not (os.path.exists(root_idx_dir + 'train_fold.csv')):
            print("[!] Splitting the data into train/val/test ...")
            # Using 5-fold cross val as used in RP-GNN paper
            cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
            k_data_splits = []
            # this is a temporary index assignment, to be used below for val splitting
            # dataset = format_dataset (dataset)
            #dataset.in_split = True
            for i in range(len(dataset.features_lists)):
                dataset[i][2].a = lambda: None
                setattr(dataset[i][2].a, 'index', i)
            a = np.random.rand(len(dataset.features_lists))
            for i in range(len(a)):
                a[i] = 1
            f_train_w = []
            f_val_w = []
            f_test_w = []
            for indexes in cross_val_fold.split(dataset.features_lists, a):
                remain_index, test_index = indexes[0], indexes[1]

                remain_set = format_dataset([dataset[index] for index in remain_index],in_split=True)

                # Gets final 'train' and 'val'
                train, val, _, __ = train_test_split(remain_set,
                                                     range(len(remain_set.feature_lists)),
                                                     test_size=0.25)
                train, val = format_dataset(train, in_split=True), format_dataset(val, in_split=True)
                test = format_dataset([dataset[index] for index in test_index], in_split=True)
                # Extracting only idxs
                idx_train = [item[2].a.index for item in train]
                idx_val = [item[2].a.index for item in val]
                idx_test = [item[2].a.index for item in test]

                '''f_train_w = csv.writer(open(root_idx_dir + dataset.name + '_train.csv', 'a+'))
                f_val_w = csv.writer(open(root_idx_dir + dataset.name + '_val.csv', 'a+'))
                f_test_w = csv.writer(open(root_idx_dir + dataset.name + '_test.csv', 'a+'))'''
                f_train_w.append(idx_train)
                f_val_w.append(idx_val)
                f_test_w.append(idx_test)

            # reading idx from the files
            val_np = np.zeros((5, len(idx_val)-1))
            test_np = np.zeros((5, len(idx_val)-1))
            for i in range(val_np.shape[0]):
                for j in range(val_np.shape[1]):
                    test_np[i, j] = f_test_w[i][j]
                    val_np[i, j] = f_val_w[i][j]


            f_val_w = val_np
            f_test_w = test_np
            f_train_w = np.array(f_train_w)

            f_train_w.tofile(root_idx_dir + 'train_fold.csv', sep=',')
            f_val_w.tofile(root_idx_dir + 'val_fold.csv', sep=',')
            f_test_w.tofile(root_idx_dir + 'test_fold.csv', sep=',')
            print("[!] Splitting done!")
        for section in ['train', 'val', 'test']:
            all_idx[section] = np.reshape(np.loadtxt(root_idx_dir + section + "_fold.csv",
                                                     delimiter=",", dtype=int), (k_splits, -1))
        return all_idx


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, name='img', data_dir="../data", data_feat="random"):
        t0 = time.time()
        self.in_split = False
        self.name = name
        self.data_dir = data_dir
        self.data_feat = data_feat
        self.k_splits = 5
        if data_feat != 'mfcc_chroma':
            self.nbr_feature = 1
            self.size = (300,12)
        else :
            self.nbr_feature = 2
            self.size = (300, 12)
        self.nbr_classes = 15
        save_dir = self.data_dir + "/save/" + self.name
        if os.path.exists(save_dir + '/' + self.data_feat + '.pkl'):
            with open(save_dir + '/' + self.data_feat + '.pkl', "rb") as f:
                f = pickle.load(f)
                self.train = f[0]
                self.val = f[1]
                self.test = f[2]
        else:
            if self.name == 'img':
                dataset = Data2Features(data_feat=self.data_feat, data_dir=self.data_dir, name=self.name)
                self.nbr_feature = dataset.nbr_feature
                self.nbr_classes = dataset.nbr_classes
                self.size = dataset.size
            elif self.name == 'graph':
                dataset = Data2Graph(data_feat=self.data_feat, data_dir=self.data_dir, name=self.name)
                if self.data_feat != 'mfcc_chroma':
                    self.nbr_feature = 300 * 12
                    self.size = (300, 12)
                else:
                    self.nbr_feature = 300 * 24
                    self.size = (300, 24)
                self.nbr_classes = dataset.nbr_classes
            elif self.name =='graph2':
                dataset = Data2Graph2(data_feat=self.data_feat, data_dir=self.data_dir, name=self.name)
                self.nbr_feature = 300 * 12
                self.size = (300, 12)



            print("[!] Dataset: ", self.name)
            # this function splits data into train/val/test and returns the indices
            self.all_idx = get_all_split_idx(dataset, self.data_dir, self.k_splits)
            self.all = dataset
            self.train = [format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in
                          range(self.k_splits)]
            self.val = [format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in
                        range(self.k_splits)]
            self.test = [format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in
                         range(self.k_splits)]
            self._save(save_dir)

        # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.cat(labels).long()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels


    def _save(self, save_dir):
        start = time.time()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + '/' + self.data_feat + '.pkl', 'wb') as f:
            pickle.dump([self.train, self.val, self.test], f)
        print(' data saved : Time (sec):', time.time() - start)


class PytorchDataset(torch.utils.data.Dataset):
    """
        FormDataset wrapping feature list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.feature_lists = lists[0]
        self.label_lists = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def format_dataset(dataset, in_split=False):
    """
        Utility function to recover data,
        pytorch compatible format
    """
    features = [data[0] for data in dataset]
    labels = [data[1] for data in dataset]
    try:
        idx = [data[2] for data in dataset]
    except IndexError:
        in_split = False
    if in_split:
        return PytorchDataset(features, labels, idx)
    else:
        return PytorchDataset(features, labels)


