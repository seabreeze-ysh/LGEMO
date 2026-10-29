"""
Attack Logs to CSV
========================
"""

import csv

import pandas as pd

from textattack.shared import AttackedText, logger
from textattack.metrics.quality_metrics import Perplexity, USEMetric
from textattack.attack_results import FailedAttackResult, SkippedAttackResult
from .logger import Logger


class CSVLogger(Logger):
    """Logs attack results to a CSV."""

    def __init__(self, filename="results.csv", color_method="file"):
        logger.info(f"Logging to CSV at path {filename}")
        self.filename = filename
        self.color_method = color_method
        self.row_list = []
        self._flushed = True
        self.df = pd.DataFrame()
        self._ppl_metric = None
        self._use_metric = None
    def _get_metrics(self):
        """延迟加载模型，确保只有在真正记录时才占用显存"""
        if self._ppl_metric is None:
            # 这里的 Perplexity 是你提供的那个类
            self._ppl_metric = Perplexity()
        if self._use_metric is None:
            self._use_metric = USEMetric()
        return self._ppl_metric, self._use_metric
    
    def log_attack_result(self, result):
        if isinstance(result, SkippedAttackResult):
            return
        original_text, perturbed_text = result.diff_color(self.color_method)
        original_text = original_text.replace("\n", AttackedText.SPLIT_TOKEN)
        perturbed_text = perturbed_text.replace("\n", AttackedText.SPLIT_TOKEN)
        result_type = result.__class__.__name__.replace("AttackResult", "")
        avg_orig_ppl = avg_atk_ppl = avg_use_score = float('nan')
        if not isinstance(result, FailedAttackResult):
            try:
                ppl_metric, use_metric = self._get_metrics()
                
                # 关键技巧：
                # 我们不直接调用 calc_ppl，而是调用 calculate([result])
                # 这样它在内部只会把这一个文本放入列表，join 后依然是原文本，
                # 彻底避开了 1024 token 的限制。
                ppl_results = ppl_metric.calculate([result])
                use_results = use_metric.calculate([result])
                
                avg_orig_ppl = ppl_results["avg_original_perplexity"]
                avg_atk_ppl = ppl_results["avg_attack_perplexity"]
                avg_use_score = use_results["avg_attack_use_score"]
            except Exception as e:
                logger.warning(f"Could not calculate advanced metrics for sample: {e}")
        row = {
            "original_text": original_text,
            "perturbed_text": perturbed_text,
            "original_score": result.original_result.score,
            "perturbed_score": result.perturbed_result.score,
            "original_output": result.original_result.output,
            "perturbed_output": result.perturbed_result.output,
            "ground_truth_output": result.original_result.ground_truth_output,
            "num_queries": result.num_queries,
            "result_type": result_type,
            "original_ppl": avg_orig_ppl,
            "attack_ppl": avg_atk_ppl,
            "attack_use_score": avg_use_score,
        }
        self.row_list.append(row)
        self.df = pd.DataFrame.from_records(self.row_list)
        self._flushed = False

    def flush(self):
        self.df.to_csv(self.filename, quoting=csv.QUOTE_NONNUMERIC, index=False)
        self._flushed = True

    def close(self):
        # self.fout.close()
        super().close()

    def __del__(self):
        if not self._flushed:
            logger.warning("CSVLogger exiting without calling flush().")
