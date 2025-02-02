# ABCというのは抽象基底クラスを定義するためのもの
# abstractmethodは抽象規定クラスから継承するクラスを作った時に、必ず作らなければいけないメソッドを定義するもの
from abc import ABC, abstractmethod

from util.data_loader import DataLoader
from util.metric_calculator import MetricCalculator
from util.models import Dataset, RecommendResult


class BaseRecommender(ABC):
    # 継承するクラスはrecommendを使うことを強制する
    @abstractmethod
    def recommend(self, dataset: Dataset, **kwargs) -> RecommendResult:
        pass

    def run_sample(self) -> None:
        # データを取得
        mind = DataLoader()
        # 推薦する
        recommend_result = self.recommend(mind)
        # 推薦結果の評価を行う
        # mind.valid_df.category~~は真の値でrecommend_resultが予測の値
        metrics = MetricCalculator().calc(
            mind.valid_df.category.tolist(),
            recommend_result
        )
        print(metrics)
