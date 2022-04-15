import csv
import os
import pickle
import numpy as np
import torch as torch
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
random.seed(42)


class Music2Features (torch.utils.data.Dataset):
    def __init__(self, name='Simple_data', data_dir="../data", data_feat="random"):
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
            print('create new dataset')
        # TODO : prepare de features in the expected modality
        else:
            self.features = np.reshape(
                np.loadtxt(self.data_dir + '/img.csv',
                           delimiter=",", dtype=float), (-1, 3, 128, 128))
            self.nbr_sample = len(self.features[:, 0, 0, 0])
            self.nbr_feature = len(self.features[0, :, 0, 0])
            self.size = (self.features.shape[3], self.features.shape[3])
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


class Idx:
    def __init__(self,idx):
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
                                                     test_size=1/k_splits)
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
            f_train_w = np.array(f_train_w)
            f_val_w = np.array(f_val_w)
            f_test_w = np.array(f_test_w)
            f_train_w.tofile(root_idx_dir + 'train_fold.csv', sep=',')
            f_val_w.tofile(root_idx_dir + 'val_fold.csv', sep=',')
            f_test_w.tofile(root_idx_dir + 'test_fold.csv', sep=',')
            print("[!] Splitting done!")
        for section in ['train', 'val', 'test']:
            all_idx[section] = np.reshape(np.loadtxt(root_idx_dir + section + "_fold.csv",
                                                     delimiter=",", dtype=int), (k_splits, -1))
        return all_idx


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, name='Simple_data', data_dir="../data", data_feat=("random")):
        t0 = time.time()
        self.in_split = False
        self.name = name
        self.data_dir = data_dir
        self.data_feat = data_feat
        self.k_splits = 5
        save_dir = self.data_dir + "/save"
        dataset = Music2Features(data_feat=self.data_feat, data_dir=self.data_dir, name=self.name)
        self.nbr_feature = dataset.nbr_feature
        self.nbr_classes = dataset.nbr_classes
        self.size = dataset.size
        if os.path.exists(save_dir + '/' + self.name + '.pkl'):
            with open(save_dir + '/' + name + '.pkl', "rb") as f:
                f = pickle.load(f)
                self.train = f[0]
                self.val = f[1]
                self.test = f[2]
        else:
            print("[!] Dataset: ", self.name)
            # this function splits data into train/val/test and returns the indices
            self.all_idx = get_all_split_idx(dataset, self.data_dir,self.k_splits)
            self.all = dataset
            self.train = [format_dataset([dataset[idx] for idx in self.all_idx['train'][split_num]]) for split_num in
                          range(self.k_splits)]
            self.val = [format_dataset([dataset[idx] for idx in self.all_idx['val'][split_num]]) for split_num in
                        range(self.k_splits)]
            self.test = [format_dataset([dataset[idx] for idx in self.all_idx['test'][split_num]]) for split_num in
                         range(self.k_splits)]
            self._save(save_dir)



    def _save(self, save_dir):
        start = time.time()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + '/' + self.name + '.pkl', 'wb') as f:
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


