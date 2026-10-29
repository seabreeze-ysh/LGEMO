"""
mo
=======


"""

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import MOSearchMethod
from textattack.transformations import WordSwapWordNet
from textattack.transformations import WordSwapHowNet
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.pre_transformation import (InputColumnModification,RepeatModification,StopwordModification,)
from .attack_recipe import AttackRecipe
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.transformations import WordSwapWordNet
from textattack.constraints.grammaticality import PartOfSpeech

class MO2024(AttackRecipe):
    """An implementation of Probability Weighted Word Saliency from "Generating
    Natural Language Adversarial Examples through Probability Weighted Word
    Saliency", Ren et al., 2019.

    Words are prioritized for a synonym-swap transformation based on a
    combination of their saliency score and maximum word-swap
    effectiveness. Note that this implementation does not include the
    Named Entity adversarial swap from the original paper, because it
    requires access to the full dataset and ground truth labels in
    advance.

    https://www.aclweb.org/anthology/P19-1103/
    """

    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapEmbedding(max_candidates=4)
        
        
        constraints = [StopwordModification()]
        #constraints.append(WordEmbeddingDistance(min_cos_sim=0.5, compare_against_original=True))
        #constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        constraints.append(WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=True))
        # input_column_modification = InputColumnModification(["premise", "hypothesis"], {"premise"})
        # constraints.append(input_column_modification)
        #transformation = WordSwapHowNet()
        #transformation = WordSwapWordNet()
        #transformation = WordSwapWordNet(max_candidates=4)
        #constraints = [RepeatModification(), StopwordModification()]
        constraints.append(MaxWordsPerturbed(max_percent=0.25))
        # use_constraint = UniversalSentenceEncoder(
        #     threshold=0.840845057,
        #     metric="angular",
        #     compare_against_original=False,
        #     window_size=10,
        #     skip_text_shorter_than_window=True,
        # )
        # constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        # search over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = MOSearchMethod()
        return Attack(goal_function, constraints, transformation, search_method)
