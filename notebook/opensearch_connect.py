#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenSearch とやり取りするスクリプト（クラス & CLI）
- 環境変数チェック
- 接続確認
- Index 作成
- ドキュメント投入（単一／バルク）
- KNN 検索
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from opensearchpy import OpenSearch, OpenSearchException
from tqdm import tqdm

# プロジェクトルートをパスに追加
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from util.data_loader import DataLoader

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class OpenSearchManager:
    def __init__(self):
        self.config = self.load_config()
        self.client = self.get_client()
        self.index_name = 'mind-recommend'
        self.index_body = {
            "settings": {
                "index": {"knn": True, "number_of_shards": 2}
            },
            "mappings": {
                "properties": {
                    "title_abstract": {
                        "type": "knn_vector",
                        "dimension": 100,
                        "method": {"name": "hnsw"}
                    }
                }
            }
        }
        loader = DataLoader()
        mind = loader.load('small')
        self.train_df = mind.train.reset_index(drop=True)
        self.valid_df = mind.valid.reset_index(drop=True)
        # ベクトルファイル
        self.w2v_train = np.load(os.path.join(ROOT, 'data', 'w2v_train_vector.np.npy'), allow_pickle=False)
        self.w2v_valid = np.load(os.path.join(ROOT, 'data', 'w2v_valid_vector.np.npy'), allow_pickle=False)

    @staticmethod
    def load_config() -> Dict[str, str]:
        load_dotenv()
        keys = [
            'OPENSEARCH_HOST_LOCAL_IP',
            'OPENSEARCH_PORT',
            'OPENSEARCH_USER_NAME',
            'OPENSEARCH_INITIAL_ADMIN_PASSWORD',
        ]
        cfg: Dict[str, str] = {}
        missing: List[str] = []
        for k in keys:
            v = os.environ.get(k)
            if not v:
                missing.append(k)
            else:
                cfg[k] = v
        if missing:
            logging.error(f"環境変数が未設定です: {missing}")
            sys.exit(1)
        return cfg

    def get_client(self) -> OpenSearch:
        try:
            return OpenSearch(
                hosts=[{'host': self.config['OPENSEARCH_HOST_LOCAL_IP'], 'port': int(self.config['OPENSEARCH_PORT'])}],
                http_compress=True,
                http_auth=(self.config['OPENSEARCH_USER_NAME'], self.config['OPENSEARCH_INITIAL_ADMIN_PASSWORD']),
                use_ssl=True,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False
            )
        except Exception:
            logging.exception("OpenSearch クライアント生成に失敗")
            sys.exit(1)

    def info(self) -> None:
        try:
            info = self.client.info()
            logging.info(f"接続OK: {info}")
        except OpenSearchException:
            logging.exception("OpenSearch info 取得に失敗")
            sys.exit(1)

    def create_index(self) -> None:
        try:
            resp = self.client.indices.create(self.index_name, body=self.index_body, ignore=400)
            if resp.get('error'):
                logging.error(f"インデックス作成エラー: {resp['error']}")
                sys.exit(1)
            logging.info(f"インデックス作成レスポンス: {resp}")
        except Exception as e:
            logging.exception(f"インデックス作成に失敗: {e}")
            sys.exit(1)

    def bulk(self, batch_size: int = 1000) -> None:
        data: List[Dict[str, Any]] = []
        for i in tqdm(range(len(self.train_df))):
            data.append({"index": {"_index": self.index_name, "_id": i}})
            data.append({
                "news_id": self.train_df.iloc[i, :]['news_id'],
                "category": self.train_df.iloc[i, :]['category'],
                "title_abstract": self.w2v_train[i]
            })
        # 一気に５万件はきづいのて1000件ずつ送る。dataはindexとデータの2つあるから実際のデータ量x2が送られる
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            try:
                resp = self.client.bulk(batch, refresh=True)
                if resp.get('errors'):
                    logging.error(f"バルクエラー: {resp}")
                else:
                    logging.info(f"バルク投入 {i + batch_size}")
            except OpenSearchException:
                logging.exception("バルク投入に失敗")

    def knn(self, k: int, valid_index_number : int) -> Dict[str, Any]:
        search_target_vector = self.w2v_valid[valid_index_number]
        query = {
            "size": k,
            "query": {
                "knn": {
                    "title_abstract": {
                        "vector": search_target_vector,
                        "k": k
                    }
                }
            }
        }
        try:
            resp = self.client.search(body=query, index=self.index_name)
            if resp.get('error'):
                logging.error(f"KNN検索エラー: {resp['error']}")
                sys.exit(1)
            logging.info("KNN検索結果:")
            logging.info(json.dumps(resp, indent=2, ensure_ascii=False))
            return resp
        except Exception as e:
            logging.exception(f"KNN検索に失敗 : {e}")
            sys.exit(1)

    def run(self, action: str, batch_size: int, knn_k: int, search_index: int) -> None:
        if action in ('info'):
            self.info()
        if action in ('create_index'):
            self.create_index()
        if action in ('bulk'):
            self.bulk(batch_size)
        if action in ('knn'):
            self.knn(knn_k, search_index)

def main():
    parser = argparse.ArgumentParser(description="OpenSearch 操作スクリプト")
    parser.add_argument('-a', '--action',
                        choices=['info', 'create_index', 'bulk', 'knn'],
                        required=True,
                        help="実行するアクションを指定")
    parser.add_argument('--batch-size', type=int, default=1000, help="バルク投入時のバッチサイズ")
    parser.add_argument('--knn-k', type=int, default=10, help="KNN検索時のk値")
    parser.add_argument('--search-index', type=int, default=0, help="KNN検索対象となるインデックス番号")
    args = parser.parse_args()

    manager = OpenSearchManager()
    manager.run(args.action, args.batch_size, args.knn_k, args.search_index)

if __name__ == '__main__':
    main()
