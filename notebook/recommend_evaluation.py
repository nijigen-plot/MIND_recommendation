# %%
import sys

sys.path.append('../')

import numpy as np
import pandas as pd

from src.lda_content_recommender import LDAContentRecommender
from src.random_recommender import RandomRecommender
from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator

# %%
data_loader = DataLoader()
mind = data_loader.load('small')
# %%
# ランダム推薦による評価
random_recommender = RandomRecommender()
random_result = random_recommender.recommend(mind)
# %%
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    np.array(mind.valid.category.tolist()),
    random_result.recommend_result
)
print(metrics)
# %%
print(np.array(mind.valid.category.tolist())[:10])
print(random_result.recommend_result[:10])

# %%
# LDA推薦による評価
lda_recommender = LDAContentRecommender()
lda_result = lda_recommender.recommend(mind, model_load_dir="../models")
# %%
metric_calculator.calc(
    np.array(mind.valid.category.tolist()),
    np.array(lda_result.recommend_result)
)

# %%
# word2vecによる評価
