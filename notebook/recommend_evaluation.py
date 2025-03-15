# %%
import sys

sys.path.append('../')

import numpy as np
import pandas as pd

from src.lda_content_recommender import LDAContentRecommender
from src.random_recommender import RandomRecommender
from src.word2vec_content_recommender import Word2VecContentRecommender
from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator

# %%
data_loader = DataLoader()
mind = data_loader.load('small')
metric_calculator = MetricCalculator()
# %%
# ランダム推薦による評価
random_recommender = RandomRecommender()
random_result = random_recommender.recommend(mind)
# %%
metrics = metric_calculator.calc(
    np.array(mind.valid.category.tolist()),
    random_result.recommend_result
)
print(metrics)
# %%
# LDA推薦による評価
lda_recommender = LDAContentRecommender()
lda_result = lda_recommender.recommend(mind, model_load_dir="../models")
# %%
lda_metrics = metric_calculator.calc(
    np.array(mind.valid.category.tolist()),
    np.array(lda_result.recommend_result)
)
print(lda_metrics)
# %%
# word2vecによる評価
w2v_recommender = Word2VecContentRecommender()
w2v_result = w2v_recommender.recommend(mind, model_load_dir="../models")
# %%
w2v_train_vector = w2v_result.train_content_vector
w2v_valid_vector = w2v_result.valid_content_vector
np.save('../data/w2v_train_vector.np', w2v_train_vector)
np.save('../data/w2v_valid_vector.np', w2v_valid_vector)
# %%
w2v_metrics = metric_calculator.calc(
    np.array(mind.valid.category.tolist()),
    np.array(w2v_result.recommend_result)
)
print(w2v_metrics)
# %%
mind.valid
