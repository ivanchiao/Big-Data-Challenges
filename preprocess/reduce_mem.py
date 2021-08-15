# -*- coding:utf-8 -*-

"""
@author : LMC_ZC

"""


import gc
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np


def reduce_mem(df, cols):

    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in tqdm(cols):

        col_type = df[col].dtypes

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2

    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    gc.collect()

    return df


if __name__ == '__main__':

    user_action = pd.read_csv(sys.argv[1] + '/user_action.csv')
    feed_info = pd.read_csv(sys.argv[1] + '/feed_info.csv')
    feed_embeddings = pd.read_csv(sys.argv[1] + '/feed_embeddings.csv')
    test_a = pd.read_csv(sys.argv[1] + '/test_a.csv')

    user_action = reduce_mem(user_action, user_action.columns)
    feed_info = reduce_mem(feed_info, feed_info.columns)
    feed_embeddings = reduce_mem(feed_embeddings, feed_embeddings.columns)
    test_a = reduce_mem(test_a, test_a.columns)

    user_action.to_csv(sys.argv[1] + '/reduce/user_action.csv', index=False)
    feed_info.to_csv(sys.argv[1] + '/reduce/feed_info.csv', index=False)
    feed_embeddings.to_csv(sys.argv[1] + '/reduce/feed_embeddings.csv', index=False)
    test_a.to_csv(sys.argv[1] + '/reduce/test_a.csv', index=False)

