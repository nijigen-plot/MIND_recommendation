import collections
import heapq
import string
from typing import List

import gensim
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.similarities import MatrixSimilarity
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.base_recommender import BaseRecommender
from util.models import Dataset, RecommendResult

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')
# nltk.download('tagsets_json')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('stopwords')


class Word2VecContentRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        vector_size = kwargs.get("vector_size", 100)
        n_epochs = kwargs.get("n_epochs", 50)
        model_load_dir = kwargs.get("model_load_dir", None)
        train_df = dataset.train
        train_df['title_abstract'] = (train_df['title'] + ' ' + train_df['abstract']).apply(lambda x : self._remove_noise(self._lemmatize_sentence(x)))
        title_abstract_data = train_df.title_abstract.tolist()
        categories = train_df.category.tolist()
        index2category = dict(zip(
            range(len(train_df)),
            train_df.category.tolist()
        ))
        if model_load_dir is None:
            try:
                input("model_load_dir引数が指定されていません。新規でモデルを作成し../models/下へ保存します。よろしければEnterを押してください。キャンセルする場合はCtrl+Cを押してください")
            except KeyboardInterrupt:
                raise KeyboardInterrupt("処理がキャンセルされました。")
            else:
                w2v_model = gensim.models.word2vec.Word2Vec(
                    title_abstract_data,
                    vector_size=vector_size,
                    window=100,
                    sg=1,
                    hs=0,
                    epochs=n_epochs,
                    min_count=5
                )
                w2v_model.save("../models/w2v.model")
                w2v_model = gensim.models.word2vec.Word2Vec.load("../models/w2v.model")
        else:
            # model_load_dir引数がある場合はそこからモデルをロードする
            w2v_model = gensim.models.word2vec.Word2Vec.load(f'{model_load_dir}/w2v.model')

        # validについてtrainと似ている文書TOP10を抜き出し、代表カテゴリを選出する
        valid_df = dataset.valid
        valid_df['title_abstract'] = (valid_df['title'] + ' ' + valid_df['abstract']).apply(lambda x : self._remove_noise(self._lemmatize_sentence(x)))
        valid_title_abstract_data = valid_df.title_abstract.tolist()
        title_abstract_in_model = set(w2v_model.wv.key_to_index.keys())
        title_abstract_vectors = self._generate_vector(
            title_abstract_data = title_abstract_data,
            title_abstract_in_model = title_abstract_in_model,
            w2v_model=w2v_model
        )
        valid_title_abstract_vectors = self._generate_vector(
            title_abstract_data = valid_title_abstract_data,
            title_abstract_in_model = title_abstract_in_model,
            w2v_model=w2v_model
        )
        similarity_matrix = cosine_similarity(title_abstract_vectors, valid_title_abstract_vectors)
        pred_categories = []
        # 転置すればforでvalidデータ１つずつ回せる
        for sm in tqdm(similarity_matrix.T):
            # list(enumerate)することでindexを付与
            indexed_array = list(enumerate(sm))
            # 類似TOP10を持ってくる
            largest_elements = heapq.nlargest(10, indexed_array, key=lambda x: x[1])
            # index2categoryのキーと値を持ってくる
            similar_news_category = [(index2category[id], similar_value) for id, similar_value in largest_elements]
            # defaultdictでカテゴリごとのスコアを計算することで、キーの存在を気にしなくてよくなる
            category_scores = collections.defaultdict(float)
            for category, score in similar_news_category:
                category_scores[category] += score

            max_category = max(category_scores, key=category_scores.get)
            pred_categories.append(max_category)

        return RecommendResult(list(pred_categories))

    # 学習したモデルから文書ごとに出現単語ベクトルを平均して文書ユニークなベクトルを作る
    def _generate_vector(self, title_abstract_data : List, title_abstract_in_model : set, w2v_model : gensim.models.word2vec.Word2Vec) -> np.array:
        title_abstract_vectors = []
        for i, title_abstract in tqdm(enumerate(title_abstract_data)):
            # 文書ごとにtitle_abstract_in_modelに入っている単語を抜き出す
            input_title_abstract = set(title_abstract) & title_abstract_in_model
            if len(input_title_abstract) == 0:
                # 単語が無い場合、ランダムなベクトルを付与する
                vector = np.random.randn(w2v_model.vector_size)
            else:
                # 単語がある場合は、単語数が行、vector_sizeの分だけ列の行列ができているので、mean(axis=0)で平均をとってベクトルにする
                vector = w2v_model.wv[input_title_abstract].mean(axis=0)
            title_abstract_vectors.append(vector)

        return np.array(title_abstract_vectors)


    def _lemmatize_sentence(self, sentence : str) -> List:
        lemmatizer = WordNetLemmatizer()
        lemmatized_sentence = []
        # pos_tag(word_tokenize)でトークン化してからタグ付け
        # タグに応じてレマタイザー用のタグを付けなおす
        for word, tag in pos_tag(word_tokenize(sentence)):
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            # レマタイズ
            lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
        return lemmatized_sentence

    # %%
    def _remove_noise(self, tokens : List) -> List:
        stop_words = stopwords.words('english')
        clean_tokens = []

        for token in tokens:
            # 空文字じゃなくて、句読点系だけのものじゃなくて、ストップワードでもない物をtokenとする
            if (len(token) > 0) and (token not in string.punctuation) and (token.lower() not in stop_words):
                clean_tokens.append(token)

        return clean_tokens
