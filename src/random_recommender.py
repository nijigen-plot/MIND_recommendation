import numpy as np

from src.base_recommender import BaseRecommender
from util.models import Dataset, RecommendResult


class RandomRecommender(BaseRecommender):
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        # メソッド内で定義しないと結果が変わる
        np.random.seed(46)
        valid_true_category = dataset.valid.category.tolist()
        # trainのcategoryからランダムに取得し、それをpredとする
        train_category = dataset.train.category.tolist()
        valid_pred_category = np.random.choice(train_category, size=len(valid_true_category))

        return RecommendResult(list(valid_pred_category))
