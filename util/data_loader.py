import os
from typing import Tuple

import pandas as pd

from util.models import Dataset


class DataLoader:
    def __init__(self, data_path: str='../data'):
        self.data_path = data_path

    def load(self, volume = 'large') -> Dataset:
        if volume in ('small','large'):
            train_df, valid_df = self._load(volume)
            return Dataset(train_df, valid_df)
        else:
            raise ValueError("volume argument must be either 'small' or 'large'")

    def _transform(self, full_data_path : str) -> pd.DataFrame:
        news_columns = [
            'news_id',
            'category',
            'subcategory',
            'title',
            'abstract',
            'url',
            'title_entities',
            'abstract_entities'
        ]
        df = pd.read_csv(f'{full_data_path}', sep='\t', names=news_columns, header=None)
        # NaNを除外して必要なカラムだけを返す
        df.dropna(inplace=True)
        # 小文字にする
        df['title'] = df['title'].str.lower()
        df['abstract'] = df['abstract'].str.lower()
        return df[['news_id', 'category', 'title', 'abstract']]

    def _load(self, volume : str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = self._transform(full_data_path=f'{self.data_path}/MIND{volume}_train/news.tsv')
        valid_df = self._transform(full_data_path=f'{self.data_path}/MIND{volume}_dev/news.tsv')

        return train_df, valid_df
