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


# %%
@st.cache_resource()
def get_manager():
    return OpenSearchManager(strategy='word2vec'), OpenSearchManager(strategy='lda')

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
    st.title("MIND Word2Vec & LDAによるコンテンツベクトル近傍検索デモ")
    st.markdown("[MIND(Microsoft News Dataset)](https://msnews.github.io/) smallのTraining,Validation Setを活用した [Word2Vec](https://arxiv.org/pdf/1310.4546),[LDA(Latent Dirichlet Allocation)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)の予測結果比較を行えます。")
    w2v_manager, lda_manager = get_manager()
    n = st.slider("近傍検索するドキュメントを選択してください", min_value=0, max_value=len(w2v_manager.valid_df_only_exists)-1, value=0)
    st.dataframe(w2v_manager.valid_df_only_exists.iloc[n, :])

    if st.button("検索開始!"):
        w2v_result = w2v_manager.knn(k=10, valid_index_number=n)
        lda_result = lda_manager.knn(k=10, valid_index_number=n)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Word2Vec 検索結果")
            w2v_hits = w2v_result["hits"]["hits"]
            for i, wdoc in enumerate(w2v_hits, 1):
                w2v_source = wdoc["_source"]
                match_index = (w2v_manager.train_df['news_id'] == w2v_source.get('news_id')).idxmax()
                st.markdown(f"- **カテゴリ**: `{w2v_source.get('category', 'N/A')}`")
                st.markdown(f"- **ニュースID**: `{w2v_source.get('news_id', 'N/A')}`")
                st.markdown(f"- **スコア**: `{wdoc.get('_score', 0):.4f}`")
                st.markdown(f"- **タイトル**: `{w2v_manager.train_df.loc[match_index, 'title']}`")
                st.markdown(f"- **要約**: `{w2v_manager.train_df.loc[match_index, 'abstract']}`")
                st.markdown("---")

        with col2:
            st.markdown("### LDA 検索結果")
            lda_hits = lda_result["hits"]["hits"]
            for i, ldoc in enumerate(lda_hits, 1):
                lda_source = ldoc["_source"]
                match_index = (lda_manager.train_df['news_id'] == lda_source.get('news_id')).idxmax()
                st.markdown(f"- **カテゴリ**: `{lda_source.get('category', 'N/A')}`")
                st.markdown(f"- **ニュースID**: `{lda_source.get('news_id', 'N/A')}`")
                st.markdown(f"- **スコア**: `{ldoc.get('_score', 0):.4f}`")
                st.markdown(f"- **タイトル**: `{lda_manager.train_df.loc[match_index, 'title']}`")
                st.markdown(f"- **要約**: `{lda_manager.train_df.loc[match_index, 'abstract']}`")
                st.markdown("---")

    st.markdown("---")
    st.markdown("[Github Repository](https://github.com/nijigen-plot/MIND_recommendation)")

if __name__ == "__main__":
    main()
