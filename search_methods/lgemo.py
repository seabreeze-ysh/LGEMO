"""
LGEMO Search
===============

"""

import numpy as np
import random
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import lgemo
from textattack.constraints.pre_transformation.stopword_modification import StopwordModification
from textattack.search_methods import SearchMethod
from collections import defaultdict
import copy
import pickle
import math
import itertools
from tqdm import tqdm
import os
class LGEMO(SearchMethod):
    """An attack that maintains a beam of the `beam_width` highest scoring
    AttackedTexts, greedily updating the beam with the highest scoring
    transformations from the current beam.

    Args:
        goal_function: A function for determining how well a perturbation is doing at achieving the attack's goal.
        transformation: The type of transformation.

    """

    def __init__(self, max_iters=None, unchanged_iters=None):
        if max_iters is None:
            max_iters = os.environ.get("LGEMO_MAX_ITERS", 1500)

        if unchanged_iters is None:
            unchanged_iters = os.environ.get("LGEMO_UNCHANGED_ITERS", 700)

        self.max_iters = int(max_iters)
        self.unchanged_iters = int(unchanged_iters)

        self.individual_tried_positions = defaultdict(set)
        self.stopword_constraint = StopwordModification()

    
#     def __init__(self):
#         self.max_iters=1500
#         self.unchanged_iters=700
#         self.individual_tried_positions = defaultdict(set)
#         self.stopword_constraint = StopwordModification()
#         self.comparison_data = []
#         self.query_counts_to_success = []
#         self._current_query_count = 0
        
#         self.final_populations_data = []


    def count_modified_words(self, x_adv, original):
        min_len = min(len(x_adv.words), len(original.words))
        return sum(1 for i in range(min_len) if x_adv.words[i] != original.words[i])
    def select_position_with_individual_history(self, individual, search_indices_set):

            individual_key = tuple(individual.words)
            tried_positions = self.individual_tried_positions[individual_key]


            available_positions = list(search_indices_set - tried_positions)

            if not available_positions:
                selected_pos = None
                return selected_pos


            if available_positions:
                selected_pos = random.choice(available_positions)

                self.individual_tried_positions[individual_key].add(selected_pos)

                return selected_pos

            return None
    def mutate(self, x_adv, x_orig,search_indices_set):
            x_new_list = []
            score_new_list = []
            pred_new_list = []

            if search_indices_set is not None:
               replace_idx = self.select_position_with_individual_history(x_adv, search_indices_set)

               if replace_idx is not None:
                  transformations = self.get_transformations(x_adv,original_text=x_orig,changed_to_original=True,indices_to_modify=[replace_idx])


                  if len(transformations) != 0:
                      results, search_over = self.get_goal_results(transformations)
                      scores = np.array([r.score for r in results])
                      cur_best_result = results[scores.argmax()]
                      return cur_best_result, search_over

            return None, False
    def perform_search(self, initial_result):
        x_orig = initial_result.attacked_text
        text = initial_result.attacked_text

        search_indices = self.stopword_constraint._get_modifiable_indices(text)
        min_mod = len(search_indices)
        best_result = initial_result
        best_adv = None
        x_len = len(text.words)

        max_modifications = x_len // 4

        population = []
        population.append(initial_result)
        no_update_rounds = 0
        iter_rounds = 0

        while no_update_rounds < self.unchanged_iters and iter_rounds < self.max_iters:
            population_updated = False
            if best_adv is not None:
                if min_mod > 1:
                    candidates = [sol for sol in population
                                  if self.count_modified_words(sol.attacked_text, x_orig) < min_mod]
                else:
                    no_update_rounds = self.unchanged_iters
            else:
                candidates = [sol for sol in population if self.count_modified_words(sol.attacked_text, x_orig) <= max_modifications]
            parent_result = random.choice(candidates)
            parent = parent_result.attacked_text
            mutation_result, search_over= self.mutate(parent, x_orig, search_indices)
            if mutation_result is None:
                iter_rounds += 1
                no_update_rounds += 1
                continue
            current_mod = self.count_modified_words(mutation_result.attacked_text, x_orig)
            if mutation_result.goal_status == GoalFunctionResultStatus.SUCCEEDED and current_mod < min_mod:
                best_adv = mutation_result
                min_mod = current_mod

            dominated = False
            to_remove = []
            x_prime_tuple = tuple(mutation_result.attacked_text.words)
            for b in population:
                b_mod = self.count_modified_words(b.attacked_text, x_orig)
                if (b.score >= mutation_result.score and b_mod <= current_mod) and \
                        (b.score > mutation_result.score or b_mod < current_mod):
                    dominated = True
                    break
                if (mutation_result.score >= b.score and current_mod <= b_mod) and \
                        (mutation_result.score > b.score or current_mod < b_mod):
                    to_remove.append(b)
            if not dominated:
                to_remove_tuples = [tuple(a.attacked_text.words) for a in to_remove]
                population = [ind for ind in population if tuple(ind.attacked_text.words) not in to_remove_tuples]
                if x_prime_tuple not in {tuple(ind.attacked_text.words) for ind in population}:
                    population.append(mutation_result)
                    population_updated = True
            if population_updated:
                no_update_rounds = 0
            else:
                no_update_rounds += 1
            iter_rounds += 1

            if search_over:
                break

            
        if best_adv is not None:
            best_result = best_adv
        else:
            scores = np.array([r.score for r in population])
            best_result = population[scores.argmax()]


    

        return best_result
    
    



    

    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["max_iters","unchanged_iters"]