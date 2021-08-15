# -*- coding:utf-8 -*-

"""
@author : LMC_ZC

"""

import sys
import pandas as pd
import numpy as np
import pickle


class Series(object):

    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self._load()

    def _load(self):
        self.feed_info = pd.read_csv(self.data_path + '/feed_info.csv')
        self.feed_emb = pd.read_csv(self.data_path + '/feed_embeddings.csv')
        self.user_action = pd.read_csv(self.data_path + '/user_action.csv')
        self.test = pd.read_csv(self.data_path + '/test_a.csv')

    def _serialize_id(self, df_list, col_name, begin=0):
        mats = list(map(lambda x: x[col_name], df_list))
        ids = np.concatenate(mats, axis=0)
        ids = np.unique(ids)
        ids_count = ids.shape[0]

        map_ids = np.arange(begin, ids_count + begin, dtype=np.int32)
        map_ids_series = pd.Series(index=ids, data=map_ids)  # 映射表

        return map_ids_series

    def _run_serialize_id(self, df, mapping_series, col_name):
        mapped_id = mapping_series.loc[df[col_name].to_list()]
        # mapped_id = df[col_name].map(lambda x: mapping_series[x])
        df[col_name] = mapped_id.values

        return df

    def _reverse_index_values(self, df):
        return pd.DataFrame(index=df.values, data={'after reverse': df.index})

    def _have_nan(self, df, col_name):
        return False if pd.isna(df[col_name]).mean() == 0.0 else True

    def _test_nan(self):
        print('authorid:' + str(self._have_nan(self.feed_info, 'authorid')))
        print('bgm_song_id:' + str(self._have_nan(self.feed_info, 'bgm_song_id')))
        print('bgm_singer_id:' + str(self._have_nan(self.feed_info, 'bgm_singer_id')))

    def series(self):
        origin2ids_userid = self._serialize_id(
            [
                self.user_action,
                self.test,
            ],
            'userid',
            begin=0
        )

        origin2ids_feedid = self._serialize_id(
            [
                self.feed_info
            ],
            'feedid',
            begin=1
        )

        origin2ids_authorid = self._serialize_id(
            [
                self.feed_info
            ],
            'authorid',
            begin=1
        )

        origin2ids_bsong = self._serialize_id(
            [
                self.feed_info
            ],
            'bgm_song_id',
            begin=1
        )

        origin2ids_bsinger = self._serialize_id(
            [
                self.feed_info
            ],
            'bgm_singer_id',
            begin=1
        )

        origin2ids_bsong = origin2ids_bsong[~origin2ids_bsong.index.duplicated(keep='first')]
        origin2ids_bsinger = origin2ids_bsinger[~origin2ids_bsinger.index.duplicated(keep='first')]

        self.user_action = self._run_serialize_id(self.user_action, origin2ids_userid, 'userid')
        self.user_action = self._run_serialize_id(self.user_action, origin2ids_feedid, 'feedid')

        self.feed_info = self._run_serialize_id(self.feed_info, origin2ids_feedid, 'feedid')
        self.feed_info = self._run_serialize_id(self.feed_info, origin2ids_authorid, 'authorid')
        self.feed_info = self._run_serialize_id(self.feed_info, origin2ids_bsong, 'bgm_song_id')
        self.feed_info = self._run_serialize_id(self.feed_info, origin2ids_bsinger, 'bgm_singer_id')

        self.test = self._run_serialize_id(self.test, origin2ids_userid, 'userid')
        self.test = self._run_serialize_id(self.test, origin2ids_feedid, 'feedid')
        self.feed_emb = self._run_serialize_id(self.feed_emb, origin2ids_feedid, 'feedid')

        return {
            'userid_voc': origin2ids_userid.shape[0],
            'feedid_voc': origin2ids_feedid.shape[0],
            'authorid_voc': origin2ids_authorid.shape[0],
            'bgm_song_id_voc': origin2ids_bsong.shape[0],
            'bgm_singer_id_voc': origin2ids_bsinger.shape[0],
            'userid_map': origin2ids_userid,
            'feedid_map': origin2ids_feedid,
            'authorid_map': origin2ids_authorid,
            'bgm_song_id_map': origin2ids_bsong,
            'bgm_singer_id_map': origin2ids_bsinger,
        }

    def _save_series(self, df):
        index = pd.Series(df.index, name='src')
        values = pd.Series(df.values, name='des')
        return pd.concat([index, values], axis=1)

    def save(self, dict_data):
        with open(self.save_path + '/statistical.pkl', 'wb') as f:
            pickle.dump([
                dict_data['userid_voc'],
                dict_data['feedid_voc'],
                dict_data['authorid_voc'],
                dict_data['bgm_song_id_voc'],
                dict_data['bgm_singer_id_voc']], f, pickle.HIGHEST_PROTOCOL)

        self._save_series(dict_data['userid_map']).to_csv(self.save_path + '/userid_map.csv', index=False)
        self._save_series(dict_data['feedid_map']).to_csv(self.save_path + '/feedid_map.csv', index=False)
        self._save_series(dict_data['authorid_map']).to_csv(self.save_path + '/authorid_map.csv',
                                                            index=False)
        self._save_series(dict_data['bgm_song_id_map']).to_csv(self.save_path + '/bgm_song_id_map.csv',
                                                               index=False)
        self._save_series(dict_data['bgm_singer_id_map']).to_csv(self.save_path + '/bgm_singer_id_map.csv',
                                                                 index=False)

        self.user_action.to_csv(self.save_path + '/user_action.csv', index=False)
        self.feed_info.to_csv(self.save_path + '/feed_info.csv', index=False)
        self.feed_emb.to_csv(self.save_path + '/feed_embeddings.csv', index=False)
        self.test.to_csv(self.save_path + '/test_a.csv', index=False)


if __name__ == '__main__':
    D = Series(sys.argv[1], sys.argv[2])
    dict_data = D.series()
    D.save(dict_data)
