# %%
# OpenSearchにデータ入れたり入れなかったりするのを試す
# 後は基本的な接続とかね

# %%
import os

import numpy as np
from dotenv import load_dotenv
from opensearchpy import OpenSearch

load_dotenv()
# %%
host = os.environ.get('OPENSEARCH_HOST_LOCAL_IP')
port = os.environ.get('OPENSEARCH_PORT')
user = os.environ.get('OPENSEARCH_USER_NAME')
password = os.environ.get('OPENSEARCH_INITIAL_ADMIN_PASSWORD')
auth = (user, password)

# %%
client = OpenSearch(
    hosts = [{'host': host, 'port': port}],
    http_compress = True, # enables gzip compression for request bodies
    http_auth = auth,
    use_ssl = True,
    verify_certs = False,
    ssl_assert_hostname = False,
    ssl_show_warn = False
)

# %%
client.info()

# %%
index_name = 'mind-recommend'
index_body = {
    "settings": {
        "index": {
            "knn": True,
            "number_of_shards" : 2
        }
    },
    "mappings": {
        "properties": {
            "title_abstract": {
                "type": "knn_vector",
                "dimension": 100,
                "method": {
                    "name": "hnsw"
                }
        }
        }
    }
}

response = client.indices.create(index_name, body=index_body)
# %%
print('\nCreating index:')
print(response)

# %%
# データを1つ入れてみる
# %%
import sys

sys.path.append('../')

import json
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.lda_content_recommender import LDAContentRecommender
from src.random_recommender import RandomRecommender
from src.word2vec_content_recommender import Word2VecContentRecommender
from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator

# %%
data_loader = DataLoader()
mind = data_loader.load('small')

w2v_train_vector = np.load('../data/w2v_train_vector.np.npy')
train_df = mind.train
# %%
document = {
    'news_id' : train_df.iloc[0,:]['news_id'],
    'category': train_df.iloc[0,:]['category'],
    'title_abstract' : w2v_train_vector[0]
}
id = '1'
# %%
response = client.index(
    index=index_name,
    body = document,
    id = id,
    refresh = True
)

print('\nAdding document:')
print(response)
# %%
# データをまとめて入れてみる
# indexが同じ場合はUPDATEされる

data: Any = []
for i in tqdm(range(len(train_df))):
    data.append({
        "index": {"_index": index_name, "_id": i}
    })
    data.append({
        "news_id": train_df.iloc[i, :]['news_id'],
        "category": train_df.iloc[i, :]['category'],
        "title_abstract": w2v_train_vector[i]
    })
# %%
# 一気に5万件はきついので1000件ずつ送る。dataはindexとデータの2つあるから実際のデータ量x2が送られる
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i : i + batch_size]
    rc = client.bulk(batch)
    print(f"Inserted {i + batch_size} records")
# %%
# Similar Searchしてみる
w2v_valid_vector = np.load('../data/w2v_valid_vector.np.npy')
search_target_vector = w2v_valid_vector[5000]
valid_df = mind.valid
# %%
query = {
  "size": 10,
  "query": {
    "knn": {
      "title_abstract": {
        "vector": search_target_vector,
        "k": 10
      }
    }
  }
}
# %%
response = client.search(
    body=query,
    index=index_name
)
# %%
print('\nSearch results:')
print(json.dumps(response, indent=4, ensure_ascii=False))
