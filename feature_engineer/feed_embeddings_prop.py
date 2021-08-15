# -*- coding:utf-8 -*-

"""
@author : LMC_ZC

"""


import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def process_feed_embeddings(feed_embeddings, reduce_dim=None):

    feed_embeddings = feed_embeddings.sort_values(by=['feedid'])

    all_feat_embeddings = [np.array(list(map(lambda x: eval(x), d.strip(' ').split(' '))), dtype=np.float32)
                           for d in feed_embeddings['feed_embedding']]

    all_feat_embeddings = np.stack(all_feat_embeddings, axis=0)

    if reduce_dim is None:
        result = all_feat_embeddings
    else:
        pca = PCA(n_components=reduce_dim)
        pca.fit(all_feat_embeddings)
        result = pca.transform(all_feat_embeddings)

    pad = np.zeros(shape=(1, result.shape[1]), dtype=result.dtype)
    return np.concatenate([pad, result], axis=0)


def run(save_path, feed_embeddings):

    feed_emb = process_feed_embeddings(feed_embeddings)
    np.save(save_path + '/feed_embeddings_512.npy', feed_emb)


if __name__ == '__main__':
    root_path = sys.argv[1]
    feed_embeddings = pd.read_csv(root_path + '/preprocess/feed_embeddings.csv')
    run(root_path + '/process', feed_embeddings)
