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
        self.train_df = mind.train
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
            logging.info(f"インデックス作成レスポンス: {resp}")
        except OpenSearchException:
            logging.exception("インデックス作成に失敗")

    def index_single(self, doc_id: str = "1") -> None:
        if self.train_df.empty or self.w2v_train.shape[0] < 1:
            logging.warning("データ不足のため単一投入スキップ")
            return
        doc = {
            "news_id": self.train_df.iloc[0]['news_id'],
            "category": self.train_df.iloc[0]['category'],
            "title_abstract": self.w2v_train[0].tolist()
        }
        try:
            resp = self.client.index(index=self.index_name, body=doc, id=doc_id, refresh=True)
            logging.info(f"単一投入({doc_id})レスポンス: {resp}")
        except OpenSearchException:
            logging.exception("単一ドキュメント投入に失敗")

    def bulk(self, batch_size: int = 1000) -> None:
        data: List[Dict[str, Any]] = []
        for i, row in self.train_df.iterrows():
            data.append({"index": {"_index": self.index_name, "_id": str(i)}})
            data.append({
                "news_id": row['news_id'],
                "category": row['category'],
                "title_abstract": self.w2v_train[i].tolist()
            })
        total = len(data) // 2
        for i in range(0, len(data), batch_size * 2):
            batch = data[i:i + batch_size * 2]
            try:
                resp = self.client.bulk(batch, refresh=True)
                if resp.get('errors'):
                    logging.error(f"バルクエラー: {resp}")
                else:
                    logging.info(f"バルク投入 {min(i//2 + batch_size, total)} / {total}")
            except OpenSearchException:
                logging.exception("バルク投入に失敗")

    def knn(self, k: int = 10) -> None:
        if self.w2v_valid.shape[0] <= 5000:
            logging.warning("検証データ不足のためKNN検索スキップ")
            return
        vec = self.w2v_valid[5000].tolist()
        query = {
            "size": k,
            "query": {
                "knn": {
                    "title_abstract": {"vector": vec, "k": k}
                }
            }
        }
        try:
            resp = self.client.search(body=query, index=self.index_name)
            logging.info("KNN検索結果:")
            print(json.dumps(resp, indent=2, ensure_ascii=False))
        except OpenSearchException:
            logging.exception("KNN検索に失敗")

    def run(self, action: str, doc_id: str, batch_size: int, knn_k: int) -> None:
        if action in ('info', 'all'):
            self.info()
        if action in ('create_index', 'all'):
            self.create_index()
        if action in ('single', 'all'):
            self.index_single(doc_id)
        if action in ('bulk', 'all'):
            self.bulk(batch_size)
        if action in ('knn', 'all'):
            self.knn(knn_k)

def main():
    parser = argparse.ArgumentParser(description="OpenSearch 操作スクリプト")
    parser.add_argument('-a', '--action',
                        choices=['info', 'create_index', 'single', 'bulk', 'knn', 'all'],
                        required=True,
                        help="実行するアクションを指定")
    parser.add_argument('--doc-id', default="1", help="単一投入時のドキュメントID")
    parser.add_argument('--batch-size', type=int, default=1000, help="バルク投入時のバッチサイズ")
    parser.add_argument('--knn-k', type=int, default=10, help="KNN検索時のk値")
    args = parser.parse_args()

    manager = OpenSearchManager()
    manager.run(args.action, args.doc_id, args.batch_size, args.knn_k)

if __name__ == '__main__':
    main()
