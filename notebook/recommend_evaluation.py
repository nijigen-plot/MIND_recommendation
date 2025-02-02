# %%
import sys

sys.path.append('../')

import numpy as np
import pandas as pd

from src.random_recommender import RandomRecommender
from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator

# %%
data_loader = DataLoader()
mind = data_loader.load()
# %%
random_recommender = RandomRecommender()
random_result = random_recommender.recommend(mind)
metric_calculator = MetricCalculator()
metrics = metric_calculator.calc(
    np.array(mind.valid.category.tolist()),
    random_result.recommend_result
)
print(metrics)
# %%
random_result.recommend_result[:10]
