# -*- coding:utf-8 -*-

"""
@author : LMC_ZC

"""

import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


if __name__ == '__main__':

    # args
    valid = True
    emb_dim = 128
    root_path = '/zhaochen/wechat_comp'

    # load
    user_action = pd.read_csv(root_path + '/data/preprocess/user_action.csv', usecols=['userid', 'feedid', 'date_'])
    feed_info = pd.read_csv(root_path + '/data/preprocess/feed_info.csv', usecols=['feedid', 'authorid'])
    user_action = pd.merge(left=user_action, right=feed_info, on='feedid')
    
    with open(root_path+'/data/preprocess/statistical.pkl', 'rb') as f:
        userid_voc, feedid_voc, authorid_voc, bgm_song_id_voc, bgm_singer_id_voc = pickle.load(f)

    if valid is True:
        user_action = user_action[user_action['date_'] < 14].reset_index(drop=True)

    # group by userid : user -> authorid
    groupby_userid = user_action.groupby(by='userid')
    sentence_by_userid = groupby_userid['authorid'].agg(list)
    
    # word2vec model
    model_by_userid = Word2Vec(sentences=sentence_by_userid, window=5, vector_size=emb_dim, sg=0, min_count=3, workers=16)
    wv_by_userid = np.random.normal(loc=0.0, scale=0.1, size=(authorid_voc+1, emb_dim))
    wv_by_userid[model_by_userid.wv.index_to_key] = model_by_userid.wv.vectors

    if valid is True:
        np.save(root_path + '/data/process/wv_authorid_13_by_userid.npy', wv_by_userid)
    else:
        np.save(root_path + '/data/process/wv_authorid_14_by_userid.npy', wv_by_userid)

    # history embedding (userid)
    idx = sentence_by_userid.index.values
    emb = [np.mean(wv_by_userid[authorids], axis=0) for authorids in sentence_by_userid]
    emb = np.stack(emb, axis=0)
    wv_user2author_history = np.random.normal(loc=0.0, scale=0.1, size=(userid_voc, emb_dim)).astype(np.float32)
    wv_user2author_history[idx] = emb

    if valid is True:
        np.save(root_path + '/data/process/wv_user2author_history_13.npy', wv_user2author_history)
    else:
        np.save(root_path + '/data/process/wv_user2author_history_14.npy', wv_user2author_history)
