import os
from typing import Tuple

import pandas as pd

from util.models import Dataset


class DataLoader:
    def __init__(self, data_path: str='../data'):
        self.data_path = data_path

    def load(self) -> Dataset:
        train_df, valid_df, test_df = self._load()
        return Dataset(train_df, valid_df, test_df)

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

    def _load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = self._transform(full_data_path=f'{self.data_path}/MINDlarge_train/news.tsv')
        valid_df = self._transform(full_data_path=f'{self.data_path}/MINDlarge_dev/news.tsv')
        test_df = self._transform(full_data_path=f'{self.data_path}/MINDlarge_test/news.tsv')

        return train_df, valid_df, test_df
