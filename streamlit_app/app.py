#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit を使った MIND 近傍検索デモアプリ
"""
# %%
import json
import os
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../notebook')))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import streamlit as st
from opensearch_connect import OpenSearchManager

from util.data_loader import DataLoader

# %%
OSM = OpenSearchManager()

# %%
@st.cache_resource()
def get_manager():
    return OpenSearchManager()

def run_knn_cli(cli_path: str, k: int, idx: int):
    cmd = [
        sys.executable, cli_path,
        "-a", "knn",
        "--knn-k", str(k),
        "--search-index", str(idx)
    ]
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    data = json.loads(output)
    return data["hits"]["hits"]

def main():
    st.title("MIND Word2Vecによるコンテンツベクトル近傍検索デモ")
    manager = get_manager()
    n = st.slider("近傍検索するドキュメントを選択してください(見切れる場合は対象セルをダブルクリックで全文見れます。)", min_value=0, max_value=len(manager.valid_df)-1, value=0)
    st.dataframe(manager.valid_df.iloc[n, :])

    if st.button("検索開始!"):
        result = manager.knn(k=10, valid_index_number=n)
        st.markdown("### 検索結果")
        hits = result["hits"]["hits"]
        for i, doc in enumerate(hits, 1):
            source = doc["_source"]
            st.markdown(f"- **カテゴリ**: `{source.get('category', 'N/A')}`")
            st.markdown(f"- **ニュースID**: `{source.get('news_id', 'N/A')}`")
            st.markdown(f"- **スコア**: `{doc.get('_score', 0):.4f}`")
            st.markdown("---")

if __name__ == "__main__":
    main()
