import numpy as np
import random
import torch
import itertools
from scipy.special import softmax
from textattack.search_methods import SearchMethod
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.constraints.pre_transformation.stopword_modification import StopwordModification
import math
from textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder import UniversalSentenceEncoder
from textattack.constraints.semantics.sentence_encoders.sentence_encoder import get_angular_sim
import numpy as np
import copy
import os
# 必须在导入pyplot之前设置后端！服务器环境必备

from collections import defaultdict

import textattack
import transformers
from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
# 导入官方API
from textattack.shared import AbstractWordEmbedding, WordEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance

class MOSearchMethod(SearchMethod):

    def __init__(self, G=20000, output_dir="rankword_MRBERTresults", 
                 enable_txt_export=True,
                 embedding=None,
                 embedding_constraint=None,
                 include_unknown_words=True,
                 cased=False,
                 distance_metric="mse"):
        super().__init__()
        self.G = G
        self.my_inf = 9999
        self.stopword_constraint = StopwordModification()
        
        self.sim_predictor = UniversalSentenceEncoder(threshold=0.0, compare_against_original=True)
        self.output_dir = output_dir
        self.enable_txt_export = enable_txt_export
        self.distance_metric = distance_metric.lower()
        

        if embedding_constraint is not None:

            self.embedding_constraint = embedding_constraint
            self.embedding = embedding_constraint.embedding
            self.include_unknown_words = embedding_constraint.include_unknown_words
            self.cased = embedding_constraint.cased

        else:

            if embedding is None:

                self.embedding = WordEmbedding.counterfitted_GLOVE_embedding()
            else:
                self.embedding = embedding
            
            self.include_unknown_words = include_unknown_words
            self.cased = cased

        

        if self.distance_metric != "mse":
            raise ValueError("当前版本仅支持MSE距离，请将distance_metric设为'mse'")
        

        self.txt_dir = os.path.join(output_dir, "rankings")
        os.makedirs(self.txt_dir, exist_ok=True)
        

        self.sample_counter = 0
        

        self.all_candidates = []
        self.sorted_candidates = []
        self.initial_score = None
        self.original_words = None
        self.original_text = None
        
    def perform_search(self, initial_result):

        self.sent_orig = initial_result.attacked_text
        self.true_label = initial_result.ground_truth_output
        text = initial_result.attacked_text

        indices_to_modify = self.stopword_constraint._get_modifiable_indices(text)
        

        
        
       
        self.search_space_map = {}
        for idx in indices_to_modify:


            transformations = self.get_transformations(
            self.sent_orig,
            original_text=self.sent_orig,
            indices_to_modify=[idx]
            )
        

            word_list = []
            for transformed_text in transformations:
                new_word = transformed_text.words[idx]

                if new_word.lower() != self.sent_orig.words[idx].lower():
                    word_list.append(new_word)
        
            if word_list:
                self.search_space_map[idx] = list(set(word_list))

                
                

        initial_sol = self._create_solution(initial_result)
        P = [initial_sol]
        
        reached_texts = {initial_result.attacked_text.text}
        best_sol = initial_sol


        for _t in range(self.G):
            if not P:
                break
                

            if all(sum(sol.mut_tag) == 0 for sol in P):
                break


            if len(P) > 1:

                change_counts = [sol.scores[1] for sol in P]
                sample_prob = softmax(change_counts)
                cur_sol = np.random.choice(P, p=sample_prob)
            else:
                cur_sol = P[0]

            # 构建变异空间
            
            self._ensure_search_space(cur_sol)

            # 变异生成新文本
            new_attacked_texts = []
            
            # Add / Swap / Remove 逻辑
            mutation_ops = [self._mutate_add, self._mutate_swap, self._mutate_rm]
            for op_fn in mutation_ops:
                new_at = op_fn(cur_sol, reached_texts)
                if new_at:
                    new_attacked_texts.append(new_at)
                    reached_texts.add(new_at.text)

            if not new_attacked_texts:
                continue

            # 评估并更新 Pareto 前沿
            results, search_over = self.get_goal_results(new_attacked_texts)
            for res in results:
                new_sol = self._create_solution(res)
                P = self._update_pareto_front(P, new_sol)
            best_sol = P[0]
            best_sore = best_sol.scores[0]
            for sol in P:
                if sol.scores[0] > best_sore:
                    best_sore = sol.scores[0]
                    best_sol = sol

            # 更新当前最佳
            # for sol in P:
            #     if sol.result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
            #         # 如果攻击成功，优先选择修改最少的
            #         if best_sol.result.goal_status != GoalFunctionResultStatus.SUCCEEDED or \
            #            sol.scores[1] < best_sol.scores[1]:
            #             best_sol = sol
            #     elif best_sol.result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
            #         # 如果都未成功，选得分最高的
            #         if sol.scores[0] > best_sol.scores[0]:
            #             best_sol = sol
            if search_over:
                print("达到查询预算上限，停止搜索。")
                break
            # 早期退出条件：如果已成功且变异空间收敛（可选）
            if (best_sol.result.goal_status == GoalFunctionResultStatus.SUCCEEDED and 
                np.all(best_sol.reached_spaces[1]) and np.all(best_sol.reached_spaces[2])):
                #print('Best Solution Converge!')
                break
        return best_sol.result
    
#         self.sample_counter += 1
#         sample_id = f"{self.sample_counter:04d}"
        
#         # print(f"\n{'='*60}")
#         # print(f"RANKWORD 正在处理样本 {sample_id}")
#         # print(f"{'='*60}")
        
#         # 重置统计数据
#         self.reset_statistics()
        
#         # 记录初始信息
#         self.initial_score = initial_result.score
#         self.original_words = initial_result.attacked_text.words
#         self.original_text = initial_result.attacked_text.text
#         text = initial_result.attacked_text
        
#         # 第一步：获取所有可能的单单词替换候选
#         transformations = self.get_transformations(
#             text, 
#             original_text=initial_result.attacked_text, 
#             changed_to_original=True
#         )
        
#         if len(transformations) == 0:
#             #print(f"⚠️  样本 {sample_id}: 没有找到任何候选替换词")
#             return initial_result
        
#         #print(f"找到 {len(transformations)} 个候选替换词，正在评估...")
        
#         # 第二步：评估所有候选词
#         results, _ = self.get_goal_results(transformations)
        
#         # 第三步：记录每个候选词的详细信息
#         self.all_candidates = []
#         for result in results:
#             modified_indices = result.attacked_text.attack_attrs.get("modified_indices", set())
#             if not modified_indices:
#                 continue
                
#             if isinstance(modified_indices, set):
#                 modified_indices = list(modified_indices)
                
#             if len(modified_indices) != 1:
#                 continue
                
#             word_index = modified_indices[0]
#             original_word = self.original_words[word_index]
#             candidate_word = result.attacked_text.words[word_index]
#             score_difference = result.score - self.initial_score
#             cos_similarity, embedding_distance = self._calculate_embedding_metrics(original_word, candidate_word)
            
#             self.all_candidates.append({
#                 "word_index": word_index,
#                 "original_word": original_word,
#                 "candidate_word": candidate_word,
#                 "score_difference": score_difference,
#                 "cos_similarity": cos_similarity,
#                 "embedding_distance": embedding_distance,
#                 "new_score": result.score
#             })
#         valid_cos = [c for c in self.all_candidates if not np.isnan(c["cos_similarity"])]
#         #print(f"成功记录 {len(self.all_candidates)} 个候选词，其中 {len(valid_cos)} 个计算了余弦相似度与{self.distance_metric}距离")
        
#         if len(self.all_candidates) == 0:
#             #print(f"⚠️  样本 {sample_id}: 没有有效候选词")
#             return initial_result
        
#         # 第四步：按分数差值从低到高排序
#         self.sorted_candidates = sorted(self.all_candidates, key=lambda x: x["score_difference"])
#         best_text = best_sol.result.attacked_text
#         diff_indices = best_text.all_words_diff(self.sent_orig)
#         selected_word_map = {idx: best_text.words[idx] for idx in diff_indices}

#         # 2. 按位置分组收集指标，计算每个位置的50%二分中位数
#         pos_metrics = defaultdict(lambda: {"cos_list": [], "dist_list": []})
#         for cand in self.all_candidates:
#             idx = cand["word_index"]
#             if not np.isnan(cand["cos_similarity"]):
#                 pos_metrics[idx]["cos_list"].append(cand["cos_similarity"])
#             if not np.isnan(cand["embedding_distance"]):
#                 pos_metrics[idx]["dist_list"].append(cand["embedding_distance"])
        
#         pos_median = {}
#         for idx in pos_metrics:
#             pos_median[idx] = {
#                 "cos_med": np.median(pos_metrics[idx]["cos_list"]) if pos_metrics[idx]["cos_list"] else float('nan'),
#                 "dist_med": np.median(pos_metrics[idx]["dist_list"]) if pos_metrics[idx]["dist_list"] else float('nan')
#             }

#         # 3. 给每个候选词打标签：是否选中 + 相似度二分 + 距离二分
#         for cand in self.all_candidates:
#             idx = cand["word_index"]
#             cand_word = cand["candidate_word"]
            
#             # 标记是否为最终对抗样本选中的词
#             cand["is_selected"] = "是" if (idx in selected_word_map and cand_word.lower() == selected_word_map[idx].lower()) else "否"
            
#             # 余弦相似度二分：≥中位数为较高，<中位数为较低
#             if np.isnan(cand["cos_similarity"]) or idx not in pos_median or np.isnan(pos_median[idx]["cos_med"]):
#                 cand["sim_class"] = "未知"
#             else:
#                 cand["sim_class"] = "较高" if cand["cos_similarity"] >= pos_median[idx]["cos_med"] else "较低"
            
#             # MSE距离二分：≥中位数为较远，<中位数为较近
#             if np.isnan(cand["embedding_distance"]) or idx not in pos_median or np.isnan(pos_median[idx]["dist_med"]):
#                 cand["dist_class"] = "未知"
#             else:
#                 cand["dist_class"] = "较远" if cand["embedding_distance"] >= pos_median[idx]["dist_med"] else "较近"
        
#         # 第六步：自动导出TXT排名文件
#         if self.enable_txt_export:
#             txt_path = os.path.join(self.txt_dir, f"sample_{sample_id}_ranking.txt")
#             self._export_ranked_candidates_to_txt_internal(txt_path, sample_id)
        
        
        
#         print(f"✅ 样本 {sample_id} 处理完成")

       
#     def _calculate_embedding_metrics(self, original_word, candidate_word):
#         """
#         计算词嵌入指标（与官方WordEmbeddingDistance逻辑完全一致）
#         返回：(余弦相似度, 对应距离)
#         """
#         try:
#             ref_word = original_word
#             transformed_word = candidate_word
            
#             if not self.cased:
#                 ref_word = ref_word.lower()
#                 transformed_word = transformed_word.lower()
            
#             try:
#                 ref_id = self.embedding.word2index(ref_word)
#                 transformed_id = self.embedding.word2index(transformed_word)
#             except KeyError:
#                 if self.include_unknown_words:
#                     return float('nan'), float('nan')
#                 return float('nan'), float('nan')
            
#             # 1. 余弦相似度（官方原生方法）
#             cos_sim = self.embedding.get_cos_sim(ref_id, transformed_id)
#             cos_sim = round(cos_sim, 6)
#             mse_dist = self.embedding.get_mse_dist(ref_id, transformed_id)
#             distance = round(mse_dist, 6)
#             # 2. 对应距离
#             # if self.distance_metric == "cosine":
#             #     distance = round(1 - cos_sim, 6)
#             # else:
#             #     mse_dist = self.embedding.get_mse_dist(ref_id, transformed_id)
#             #     distance = round(mse_dist, 6)
            
#             return cos_sim, distance
            
#         except Exception as e:
#             print(f"⚠️  计算单词嵌入指标失败: {original_word} ↔ {candidate_word}, 错误: {str(e)}")
#             return float('nan'), float('nan')
#     def _export_ranked_candidates_to_txt_internal(self, save_path, sample_id):
#         """内部方法：导出排名信息到TXT文件"""
#         try:
#             ranked_candidates = self.sorted_candidates[::-1]
#             total = len(ranked_candidates)
#             distance_name = "MSE距离"
            
#             # 距离统计
#             valid_distances = [c["embedding_distance"] for c in ranked_candidates if not np.isnan(c["embedding_distance"])]
#             avg_distance = round(np.mean(valid_distances), 6) if valid_distances else float('nan')
#             min_distance = round(np.min(valid_distances), 6) if valid_distances else float('nan')
#             max_distance = round(np.max(valid_distances), 6) if valid_distances else float('nan')
            
#             # 余弦相似度统计
#             valid_cos = [c["cos_similarity"] for c in ranked_candidates if not np.isnan(c["cos_similarity"])]
#             avg_cos = round(np.mean(valid_cos), 6) if valid_cos else float('nan')
#             min_cos = round(np.min(valid_cos), 6) if valid_cos else float('nan')
#             max_cos = round(np.max(valid_cos), 6) if valid_cos else float('nan')

#             # 最终选中词数量统计
#             selected_count = sum(1 for c in ranked_candidates if c["is_selected"] == "是")
            
#             with open(save_path, 'w', encoding='utf-8') as f:
#                 f.write("="*165 + "\n")
#                 f.write(f"RANKWORD 候选词排名报告（余弦相似度 + {distance_name} + 选中标记 + 二分分类）\n")
#                 f.write(f"样本编号: {sample_id}\n")
#                 f.write("="*165 + "\n")
#                 f.write(f"原始文本: {self.original_text}\n")
#                 f.write(f"原始置信度分数: {self.initial_score:.6f}\n")
#                 f.write(f"总候选词数量: {total} | 最终选中替换词数量: {selected_count}\n")
#                 f.write(f"有效余弦相似度数量: {len(valid_cos)}\n")
#                 f.write(f"平均余弦相似度: {avg_cos} | 最小: {min_cos} | 最大: {max_cos}\n")
#                 f.write(f"平均{distance_name}: {avg_distance} | 最小: {min_distance} | 最大: {max_distance}\n")
#                 f.write("="*165 + "\n")
#                 # 表头
#                 header = (f"{'排名':<6}"
#                           f"{'原单词':<15}"
#                           f"{'候选词':<15}"
#                           f"{'置信度差值':<15}"
#                           f"{'余弦相似度':<12}"
#                           f"{distance_name:<12}"
#                           f"{'是否选中':<8}"
#                           f"{'相似度分类':<12}"
#                           f"{'距离分类':<12}"
#                           f"{'单词位置':<8}\n")
#                 f.write(header)
#                 f.write("-"*165 + "\n")
                
#                 for rank, candidate in enumerate(ranked_candidates, start=1):
#                     line = (f"{rank:<6d}"
#                             f"{candidate['original_word']:<15}"
#                             f"{candidate['candidate_word']:<15}"
#                             f"{candidate['score_difference']:<15.6f}"
#                             f"{str(candidate['cos_similarity']):<12}"
#                             f"{str(candidate['embedding_distance']):<12}"
#                             f"{candidate['is_selected']:<8}"
#                             f"{candidate['sim_class']:<12}"
#                             f"{candidate['dist_class']:<12}"
#                             f"{candidate['word_index']:<8d}\n")
#                     f.write(line)
                
#                 # 底部规则说明
#                 f.write("\n" + "="*165 + "\n")
#                 f.write("分类规则说明：\n")
#                 f.write("1. 基于同位置所有候选词的中位数（50%二分位）进行分类\n")
#                 f.write("2. 相似度「较高」：余弦相似度 ≥ 同位置中位数，语义与原词更接近\n")
#                 f.write("3. 距离「较远」：MSE距离 ≥ 同位置中位数，语义与原词差异更大\n")
#                 f.write("4. 「未知」表示候选词不在词嵌入词汇表中，无法计算对应指标\n")
#                 f.write("5. 「是否选中」标记该候选词是否出现在最终生成的对抗样本中\n")
            
#             print(f"📄 排名文件已保存: {save_path}")
#             return True
#         except Exception as e:
#             print(f"❌ 导出TXT失败: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return False

#     def get_embedding_statistics(self):
#         """获取当前样本的词嵌入综合统计信息"""
#         if not self.all_candidates:
#             return {}
            
#         valid_cos = [c["cos_similarity"] for c in self.all_candidates if not np.isnan(c["cos_similarity"])]
#         valid_distances = [c["embedding_distance"] for c in self.all_candidates if not np.isnan(c["embedding_distance"])]
        
#         stats = {
#             "total_count": len(self.all_candidates),
#             "valid_cos_count": len(valid_cos),
#             "valid_distance_count": len(valid_distances),
#             "distance_metric": self.distance_metric
#         }
        
#         if valid_cos:
#             stats.update({
#                 "cos_mean": round(np.mean(valid_cos), 6),
#                 "cos_median": round(np.median(valid_cos), 6),
#                 "cos_max": round(np.max(valid_cos), 6),
#                 "cos_min": round(np.min(valid_cos), 6),
#                 "cos_std": round(np.std(valid_cos), 6)
#             })
        
#         if valid_distances:
#             stats.update({
#                 "distance_mean": round(np.mean(valid_distances), 6),
#                 "distance_median": round(np.median(valid_distances), 6),
#                 "distance_max": round(np.max(valid_distances), 6),
#                 "distance_min": round(np.min(valid_distances), 6),
#                 "distance_std": round(np.std(valid_distances), 6)
#             })
            
#         return stats
    # def reset_statistics(self):
    #     """重置所有统计数据"""
    #     self.all_candidates.clear()
    #     self.sorted_candidates.clear()
    #     self.initial_score = None
    #     self.original_words = None
    #     self.original_text = None
    def _create_solution(self, result):
    
    
        is_success = (result.goal_status == GoalFunctionResultStatus.SUCCEEDED)
    
  
        if is_success:
            attack_score = self.my_inf 
        else:
            attack_score = result.score
    
    
        diff_count = len(result.attacked_text.all_words_diff(self.sent_orig))
        if diff_count == 0:
            sim_score = 1.0
        else:
            embeddings = self.sim_predictor.encode([result.attacked_text.text, self.sent_orig.text])
            emb1 = torch.tensor(embeddings[0]).unsqueeze(dim=0)
            emb2 = torch.tensor(embeddings[1]).unsqueeze(dim=0)
            
            sim_score = get_angular_sim(emb1, emb2).item()
        sol = type('obj', (object,), {})() 
        sol.result = result
        sol.sim = sim_score
        sol.scores = [attack_score, diff_count] 
        sol.mut_tag = np.ones(3)
        sol.search_space = [None, None, None]
        sol.reached_spaces = [[False], [False], [False]]
        return sol

    def _update_pareto_front(self, P, new_sol):
        """Pareto 支配更新逻辑"""
        add_new = True
        for exist_sol in P:
            is_dom = (exist_sol.scores[0] >= new_sol.scores[0] and exist_sol.scores[1] <= new_sol.scores[1])
            is_strict_dom = is_dom and (exist_sol.scores[0] > new_sol.scores[0] or exist_sol.scores[1] < new_sol.scores[1])
            if is_strict_dom:
                add_new = False
                break
            if new_sol.result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                is_same_score = (exist_sol.scores[0] == new_sol.scores[0] and exist_sol.scores[1] == new_sol.scores[1])
                
                # 当两维多目标分数完全等价时，启动第三维度语义相似度质检：
                # 如果旧解的语义相似度(sim)严格好于新解，说明新解是低质的冗余样本，直接拦截丢弃
                if is_same_score and exist_sol.sim > new_sol.sim:
                    add_new = False
                    break
                    
            # if exist_sol.scores[0] >= new_sol.scores[0] and exist_sol.scores[1] <= new_sol.scores[1]:
            #     # if exist_sol.scores[0] > new_sol.scores[0] or exist_sol.scores[1] < new_sol.scores[1]:
            #     add_new = False
            #     break
        
        if add_new:
            # 移除被新解支配的旧解
            P_new = []
            for exist_sol in P:
                is_dom = (new_sol.scores[0] >= exist_sol.scores[0] and new_sol.scores[1] <= exist_sol.scores[1])
                if not is_dom:
                    P_new.append(exist_sol)
            P_new.append(new_sol)
            return P_new
        else:
            return P
            # P = [s for s in P if not (
            #     new_sol.scores[0] >= s.scores[0] and new_sol.scores[1] <= s.scores[1] and 
            #     (new_sol.scores[0] > s.scores[0] or new_sol.scores[1] < s.scores[1])
            # )]
            # P = [s for s in P if not (new_sol.scores[0] >= s.scores[0] and new_sol.scores[1] <= s.scores[1])]
            # P.append(new_sol)
        

    def _ensure_search_space(self, sol):
        if sol.search_space[0] is not None:
            return

    # 1. 找出当前解相对于原句改动了哪些位置
        diff_indices = sol.result.attacked_text.all_words_diff(self.sent_orig)
        all_indices = list(self.search_space_map.keys())
    
        sol.search_space = [[], [], []] # Add, Swap, Rm
    
        for idx in all_indices:
            if idx in diff_indices:
            # --- 已经改过的位置 ---
            # Swap 空间：该位置可以换的其他候选词
                sol.search_space[1].extend([(idx, w) for w in self.search_space_map[idx]])
            # Rm 空间：该位置可以被恢复
                sol.search_space[2].append(idx)
            else:
            # --- 还没改过的位置 ---
            # Add 空间：如果还没成功，可以尝试修改这里
                if sol.result.goal_status != GoalFunctionResultStatus.SUCCEEDED:
                    sol.search_space[0].extend([(idx, w) for w in self.search_space_map[idx]])

    # 2. 打乱顺序增加随机性
        for i in range(3):
            if not sol.search_space[i]:
                sol.mut_tag[i] = 0
            else:
                random.shuffle(sol.search_space[i])
                sol.reached_spaces[i] = np.zeros(len(sol.search_space[i]), dtype=bool)
    

    def _mutate_add(self, sol, reached):
        if sol.mut_tag[0] == 0: return None
    
        diff_indices = sol.result.attacked_text.all_words_diff(self.sent_orig)
        current_diff_count = len(diff_indices)
        max_percent = 0.25
        min_num_words = min(len(sol.result.attacked_text.words), len(self.sent_orig.words))
        max_words_perturbed = math.ceil(min_num_words * max_percent)
        if (current_diff_count + 1) > max_words_perturbed:
        # 标记该解的 Add 变异空间已关闭，因为它已经达到了法律上限
            sol.mut_tag[0] = 0 
            return None
        cur_space = sol.search_space[0]
        cur_rnd_indices = list(range(len(cur_space)))
        random.shuffle(cur_rnd_indices)
    # sol.search_space[0] 此时存的是 [(idx1, wordA), (idx2, wordB), ...]
        for idx_in_space in cur_rnd_indices:
            idx, new_word = cur_space[idx_in_space]
            
            # 【核心】：在 sol 当前的文本基础上修改，实现 L0 增加
            new_at = sol.result.attacked_text.replace_word_at_index(idx, new_word)
            
            if new_at.text not in reached:
                return new_at
        
    
            
    # 如果循环走完都没找到没去过的样本，标记该解的 Add 空间耗尽
        sol.mut_tag[0] = 0
        return None

    def _mutate_swap(self, sol, reached):
    
        if sol.mut_tag[1] == 0: return None
        cur_space = sol.search_space[1]
        cur_rnd_indices = list(range(len(cur_space)))
        random.shuffle(cur_rnd_indices)
        
        for idx_in_space in cur_rnd_indices:
            
            if sol.result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                sol.reached_spaces[1][idx_in_space] = True
            idx, new_word = cur_space[idx_in_space]
            
            # 排除掉当前位置已经在用的那个词，避免原地踏步
            if new_word.lower() == sol.result.attacked_text.words[idx].lower():
                continue
                
            new_at = sol.result.attacked_text.replace_word_at_index(idx, new_word)
            
            if new_at.text not in reached:
                return new_at
            
        sol.mut_tag[1] = 0
        return None

    def _mutate_rm(self, sol, reached):
   
        if sol.mut_tag[2] == 0: return None
        cur_space = sol.search_space[2]
        cur_rnd_indices = list(range(len(cur_space)))
        random.shuffle(cur_rnd_indices)
        
        for idx_in_space in cur_rnd_indices:
            if sol.result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                sol.reached_spaces[2][idx_in_space] = True
            idx = cur_space[idx_in_space]
            
            # 从原始文本对象 self.sent_orig 中取回原词
            orig_word = self.sent_orig.words[idx]
            
            # 在当前解基础上把词换回去
            new_at = sol.result.attacked_text.replace_word_at_index(idx, orig_word)
            
            if new_at.text not in reached:
                return new_at
            
        sol.mut_tag[2] = 0
        return None
    @property
    def is_black_box(self):
        return True

    def extra_repr_keys(self):
        return ["G"]