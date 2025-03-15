import dataclasses
from typing import List

import numpy as np
import pandas as pd


# データ読む。test使わないかもしれん
@dataclasses.dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    valid: pd.DataFrame

# 予測結果を返す
@dataclasses.dataclass(frozen=True)
class RecommendResult:
    recommend_result : List[str]
    train_content_vector : np.array
    valid_content_vector : np.array

# 予測結果と真のデータをもとに評価値を返してくれるやつ
@dataclasses.dataclass(frozen=True)
class Metrics:
    accuracy: np.float64
    macro_precision: np.float64
    macro_recall: np.float64

    def __repr__(self):
        return f'Accuracy={self.accuracy:.3f}, Macro Precision={self.macro_precision:.3f}, Macro Recall={self.macro_recall:.3f}'
