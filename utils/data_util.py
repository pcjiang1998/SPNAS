# coding: utf-8

import pickle

import numpy as np


def save_obj(obj, pkl_file):
    """
    save obj in file
    """
    with open(pkl_file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(pkl_file):
    """
    load obj from file
    """
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)


def split_dataset(data, target):
    """
    split dataset as 4:1
    """
    assert isinstance(data, np.ndarray) and isinstance(target, np.ndarray), 'data and target should be np.ndarray'
    assert data.shape[0] == target.shape[0], 'data and target should have the same number of instances'
    # data,target=torch.from_numpy(data).float(),torch.from_numpy(target).long()
    indices = list(range(len(data)))
    split = int(np.floor(0.2 * len(data)))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    return {'data': data[train_idx], 'target': target[train_idx]}, {'data': data[valid_idx],
                                                                    'target': target[valid_idx]}


def pca(data, dim=2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim)
    data_pca = pca.fit_transform(data)
    return data_pca
