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


class LDAContentRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        factors = kwargs.get("factors", 50)
        n_epochs = kwargs.get("n_epochs", 30)
        model_load_dir = kwargs.get("model_load_dir", None)
        train_df = dataset.train
        train_df['title_abstract'] = (train_df['title'] + ' ' + train_df['abstract']).apply(lambda x : self._remove_noise(self._lemmatize_sentence(x)))
        common_dictionary = Dictionary(train_df['title_abstract'].tolist())
        common_corpus = [common_dictionary.doc2bow(text) for text in train_df["title_abstract"].tolist()]
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
                # モデルを作成して保存する
                lda_model = gensim.models.LdaModel(
                    common_corpus,
                    id2word=common_dictionary,
                    num_topics=factors,
                    passes=n_epochs
                )
                lda_model.save("../models/lda.model")
                lda_model = gensim.models.LdaModel.load("../models/lda.model")
        else:
            # model_load_dir引数がある場合はそこからモデルをロードする
            lda_model = gensim.models.LdaModel.load(f'{model_load_dir}/lda.model')


        # validについてtrainと似ている文書TOP10を抜き出し、代表カテゴリを選出する
        valid_df = dataset.valid
        valid_df['title_abstract'] = (valid_df['title'] + ' ' + valid_df['abstract']).apply(lambda x : self._remove_noise(self._lemmatize_sentence(x)))
        valid_corpus = [common_dictionary.doc2bow(text) for text in valid_df["title_abstract"].tolist()]
        # 文書 x コーパス数でコサイン類似度が出せる行列を作る
        index = MatrixSimilarity(lda_model[common_corpus])
        pred_categories = []
        # 予測対象の文書1つ1つの代表カテゴリ予測結果を求める
        for vc in tqdm(valid_corpus):
            topic_distribution = lda_model[vc]
            sims = index[topic_distribution]
            # similar_docs = sorted(enumerate(sims), key=lambda item: -item[1])[:10]
            # 単純にソートして探すよりもこちらのほうが計算効率が良い。
            similar_docs = heapq.nlargest(10, enumerate(sims), key=lambda item: item[1])
            similar_news_category = [(index2category[id], similar_value) for id, similar_value in similar_docs]
            # defaultdictでカテゴリごとのスコアを計算することで、キーの存在を気にしなくてよくなる
            category_scores = collections.defaultdict(float)
            for category, score in similar_news_category:
                category_scores[category] += score

            max_category = max(category_scores, key=category_scores.get)
            pred_categories.append(max_category)

        # train,validのベクトルを作成
        train_vectors = [
            self._sparse_to_dense(lda_model[doc_bow], factors)
            for doc_bow in common_corpus
        ]
        valid_vectors = [
            self._sparse_to_dense(lda_model[doc_bow], factors)
            for doc_bow in valid_corpus
        ]

        return RecommendResult(list(pred_categories), np.array(train_vectors), np.array(valid_vectors))

    def _sparse_to_dense(self, topic_dist, num_topics):
        vec = [0.0] * num_topics
        for topic_id, prob in topic_dist:
            vec[topic_id] = prob
        return vec

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
