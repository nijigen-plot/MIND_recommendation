from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from util.models import Metrics


class MetricCalculator:
    def calc(
        self,
        true_category: NDArray[np.str_],
        pred_category: NDArray[np.str_]
    ) -> Metrics:

        accuracy, precision, recall = self._calc_accuracies(true_category, pred_category)

        return Metrics(accuracy, precision, recall)

    def _calc_accuracies(self, true_category: NDArray[np.str_], pred_category: NDArray[np.str_]) -> Tuple[np.float64, np.float64, np.float64]:
        df = pd.DataFrame({
            'True': true_category,
            'Pred': pred_category
        })
        # True,PredでGroupbyしてUnstackで横持ちにする
        conf_matrix = df.groupby(['True', 'Pred']).size().unstack(fill_value=0)
        all = conf_matrix.sum().sum()
        # TP, FN, FPの値は行や列ごとに精度をだして平均する感じになる
        # 対角線にあるのはTrue Positive
        tp_vector = np.diag(conf_matrix)
        # 行レベルで足し上げるのはFN
        fn_vector = conf_matrix.sum(axis=1) - tp_vector
        # 列レベルで足し上げるのはFP
        fp_vector = conf_matrix.sum(axis=0) - tp_vector

        # %%
        # 精度は全体分のTP
        accuracy = tp_vector.sum() / all
        # Precision,Recallについては行or列レベルで出した後に平均をとる
        precision = np.where((tp_vector + fp_vector) != 0, tp_vector / (tp_vector + fp_vector), np.nan)
        recall = np.where((tp_vector + fn_vector) != 0, tp_vector / (tp_vector + fn_vector), np.nan)
        avg_precision = np.nanmean(precision)
        avg_recall = np.nanmean(recall)

        return (accuracy, avg_precision, avg_recall)
