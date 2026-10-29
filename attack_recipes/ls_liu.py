#from textattack.constraints.pre_transformation import (StopwordModification)
from textattack.constraints.pre_transformation import (InputColumnModification,RepeatModification,StopwordModification,)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import LS
from textattack.transformations import WordSwapEmbedding
from textattack.transformations import WordSwapWordNet
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack import Attack
from textattack.constraints.grammaticality import PartOfSpeech
from .attack_recipe import AttackRecipe
from textattack.transformations import WordSwapHowNet
from textattack.constraints.grammaticality.language_models import (
    Google1BillionWordsLanguageModel,
)
class LS2022(AttackRecipe):
    """A recipe for building an NLP adversarial attack from the literature."""

    @staticmethod
    def build(model):
        """Creates pre-built :class:`~textattack.Attack` that correspond to
        attacks from the literature.

        Args:
            model_wrapper (:class:`~textattack.models.wrappers.ModelWrapper`):
                :class:`~textattack.models.wrappers.ModelWrapper` that contains the victim model and tokenizer.
                This is passed to :class:`~textattack.goal_functions.GoalFunction` when constructing the attack.
            kwargs:
                Additional keyword arguments.
        Returns:
            :class:`~textattack.Attack`
        """
        #transformation = WordSwapEmbedding(max_candidates=4)
        #transformation = WordSwapWordNet()
        #transformation = WordSwapHowNet()
        transformation = WordSwapEmbedding(max_candidates=50)
        
        #goal_function = MinimizeBleu(model_wrapper)
        #transformation = WordSwapInflections()
        constraints = [StopwordModification()]
        #constraints.append(WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=True))
        # input_column_modification = InputColumnModification(["premise", "hypothesis"], {"premise"})
        # constraints.append(input_column_modification)
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.5, compare_against_original=True))
        constraints.append(MaxWordsPerturbed(max_percent=0.25))
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        #constraints.append(Google1BillionWordsLanguageModel(top_n_per_index=4, compare_against_original=False))
        #constraints.append(WordEmbeddingDistance(max_mse_dist=0.5, compare_against_original=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.840845057,
            metric="angular",
            compare_against_original=False,
            window_size=10,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model)
        search_method = LS()
        return Attack(goal_function, constraints, transformation, search_method)