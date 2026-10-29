"""
LS Search
===============

"""

import numpy as np
import copy
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod


class LS(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.

    """

    def __init__(self):
        self.best_score = 0
        self._current_query_count = 0

    def perform_search(self, initial_result):
        text = initial_result.attacked_text
        best_result = initial_result
        len_text = len(text.words)
        
        self._current_query_count = 0
        first_success_query_count = None
        while not best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            transformations = self.get_transformations(
                    text, original_text=initial_result.attacked_text, changed_to_original=True
                )
            
            '''for i in range(len_text):
                if text.words[i] != initial_result.attacked_text.words[i]:
                    w = initial_result.attacked_text.words[i]
                    transformations.append(text.replace_word_at_index(i, w))
            print(f"【最终长度】changed_to_original=True，transformed_texts 长度：{len(transformations)}")'''
            if len(transformations) == 0:
                # If we did not find any possible perturbations, give up.
                return best_result
            results, search_over = self.get_goal_results(transformations)
            self._current_query_count += len(transformations)
            scores = np.array([r.score for r in results])
            cur_best_result = results[scores.argmax()]
            if first_success_query_count is None and \
                cur_best_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                first_success_query_count = self._current_query_count
            if cur_best_result.score > best_result.score:
                best_result = cur_best_result
                text = best_result.attacked_text
            else:
                break
            if search_over:
                break
        is_set_valid = True
        for i in range(len_text):
            if text.words[i] != initial_result.attacked_text.words[i]:
                transformations = self.get_transformations(text, original_text=initial_result.attacked_text,indices_to_modify=[i])
                if  len(transformations) > 0:
                    is_set_valid = False
                    break
            else:
                transformations = self.get_transformations(text, original_text=initial_result.attacked_text,
                                                           indices_to_modify=[i])
                if len(transformations) > 1:
                    is_set_valid = False
                    break
        if is_set_valid:
            #x_supp = initial_result.attacked_text.copy()
            x_supp = copy.deepcopy(initial_result.attacked_text)
            for i in range(len_text):
                if text.words[i] != initial_result.attacked_text.words[i]:
                    w = initial_result.attacked_text.words[i]
                    x_supp = x_supp.replace_word_at_index(i, w)
                else:
                    transformations = self.get_transformations(
                    x_supp, original_text=initial_result.attacked_text, indices_to_modify=[i]
                )
                    if len(transformations) == 0:
                        continue
                    x_supp = transformations[0]

            results, search_over = self.get_goal_results([x_supp])
            sup_result = results[0]
            if sup_result.score > best_result.score:
               best_result = sup_result


        return best_result

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["best_score"]
