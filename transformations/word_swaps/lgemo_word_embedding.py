"""
Custom Word Swap by Embedding
-------------------------------
规则：
1. 初始召回50个词向量近邻
2. 过滤保留 MSE 距离 ≤ 0.5 的候选
3. 按余弦相似度从小到大排序，取前4个最终候选
"""

from textattack.shared import AbstractWordEmbedding, WordEmbedding
from .word_swap import WordSwap


class WordSwapEmbeddingCustom(WordSwap):
    """Transforms an input by replacing its words with synonyms in the word
    embedding space (custom MSE filter + cos sort rule).

    Args:
        initial_candidates (int): 初始从词空间召回的近邻数量，默认50
        max_candidates (int): 最终返回的候选词数量，默认4
        mse_threshold (float): MSE距离最大阈值，默认0.5
        embedding (textattack.shared.AbstractWordEmbedding): Wrapper for word embedding
    """

    def __init__(
        self,
        initial_candidates=50,
        max_candidates=4,
        mse_threshold=0.5,
        embedding=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if embedding is None:
            embedding = WordEmbedding.counterfitted_GLOVE_embedding()
        
        self.initial_candidates = initial_candidates
        self.max_candidates = max_candidates
        self.mse_threshold = mse_threshold
        
        if not isinstance(embedding, AbstractWordEmbedding):
            raise ValueError(
                "`embedding` object must be of type `textattack.shared.AbstractWordEmbedding`."
            )
        self.embedding = embedding

    def _get_replacement_words(self, word):
        try:
            word_lower = word.lower()
            word_id = self.embedding.word2index(word_lower)

            # 步骤1：初始召回50个近邻候选
            nnids = self.embedding.nearest_neighbours(word_id, self.initial_candidates)

            valid_candidates = []
            for nbr_id in nnids:
                # 步骤2：MSE距离过滤，仅保留≤0.5的候选
                mse_val = self.embedding.get_mse_dist(word_id, nbr_id)
                if mse_val > self.mse_threshold:
                    continue
                
                # 计算余弦相似度，用于后续排序
                cos_val = self.embedding.get_cos_sim(word_id, nbr_id)
                nbr_word = self.embedding.index2word(nbr_id)
                valid_candidates.append((nbr_word, cos_val))

            # 步骤3：按余弦相似度从小到大排序，取前max_candidates个
            valid_candidates.sort(key=lambda x: x[1])
            n = len(valid_candidates)
            if n <= self.max_candidates:
            # 候选不足4个，直接返回全部
                top_candidates = valid_candidates
            else:
            # 计算中间段的起始索引，取中间max_candidates个
                start_idx = (n - self.max_candidates) // 2
                top_candidates = valid_candidates[start_idx : start_idx + self.max_candidates]
            #top_candidates = valid_candidates[: self.max_candidates]

            # 还原原单词大小写格式后返回
            return [recover_word_case(w, word) for w, _ in top_candidates]

        except KeyError:
            # 单词不在词表中，返回空列表
            return []

    def extra_repr_keys(self):
        return ["initial_candidates", "max_candidates", "mse_threshold", "embedding"]


def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.
    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        return word