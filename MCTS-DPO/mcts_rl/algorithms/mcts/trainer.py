from __future__ import annotations

from typing import Any

import random
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from mcts_rl.datasets import (
    PromptOnlyBatch, PromptOnlyPostBatch,
)
from mcts_rl.trainers import TSRLTrainer
from mcts_rl.utils import (
    gather_log_probabilities,
    get_all_reduce_max,
    get_all_reduce_mean,
    math_equal,
    extract_answer,
    csr_equal,
    calculate_preference_confidence,
    get_final_qa_index,
    pad_tensors,
)
from mcts_rl.configs.constants import PROMPT_ASSISTANT, PROMPT_BEGIN, IGNORE_INDEX
from mcts_rl.algorithms.mcts.mcts import (
    StepLMWorldModel, 
    StepLMConfig, 
    SearchArgs, 
    MCTS,
    MCTSWDPOP,
    MCTSNode,
    MCTSConfig, 
    TreeConstructor,
)


class MCTSTrainer(TSRLTrainer):
    TRAINING_TYPE = 'mcts'

    def init_mcts_searcher(self) -> None:
        world_model = StepLMWorldModel(
            max_length=self.generation_config.max_length,
            base_tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )
        search_cfg = StepLMConfig(SearchArgs(
            ref_policy_model=self.actor_reference_model,
            base_tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            n_actions=self.args.n_actions,
            n_init_actions=self.args.n_init_actions,
            breadth_limit=self.args.breadth_limit,
            depth_limit=self.args.depth_limit,
            force_terminating_on_depth_limit=self.args.force_terminating_on_depth_limit,
            kl_coeff=self.args.kl_coeff,
            disable_tqdm=False,
            no_self_eval=self.args.no_self_eval,
            reward_model=self.reward_model if self.use_reward_model else None,
            reward_tokenizer=self.reward_tokenizer if self.use_reward_model else None,
            use_code=self.args.use_code,
            use_mcq=self.args.use_mcq,
            eval_mode=self.args.eval_mode,
            temperature=self.args.temperature,
            init_temperature=self.args.init_temperature,
            get_tp_zero=self.args.get_tp_zero,
            model_type=self.args.model_type,
            include_gt=(not self.args.not_include_gt),
            verbose=self.args.verbose,
        ))
        mcts_algo = MCTS(MCTSConfig(
            w_exp=self.args.w_exp,
            depth_limit=self.args.depth_limit,
            breadth_limit=self.args.breadth_limit,
            n_iters=self.args.n_iters,
            temperature=self.args.mcts_temperature,
            temperature_decay_ratio=self.args.mcts_temperature_decay_ratio,
            consider_diversity=(not self.args.no_consider_diversity),
            length_penalty=self.args.mcts_length_penalty,
        ))
        self.mcts_searcher = TreeConstructor(
            world_model=world_model, 
            search_config=search_cfg, 
            search_algo=mcts_algo,
        )
    
    def tree_constructor(self, prompt_only_batch: PromptOnlyBatch | PromptOnlyPostBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch['input_ids']
        attention_mask = prompt_only_batch['attention_mask']
        answer = prompt_only_batch['answer']
        assert input_ids.size(0) == 1, '''Only support one instance per device.'''
        seq, attn_msk = input_ids[0], attention_mask[0]
        gt_answer, solution = answer[0], prompt_only_batch['reasoning'][0]
        
        # if solution.strip():
        #     self.mcts_searcher.search_config.generation_config.max_new_tokens = min(
        #         self.args.max_new_tokens,
        #         max(self.generation_config.max_new_tokens // 4,
        #             len(self.tokenizer.encode(solution)) // max(1, self.args.depth_limit - 1))
        #     )
        
        self.mcts_searcher.search_config.use_code = ('\nprint(' in solution)
        if self.mcts_searcher.search_algo.policy_model is None or self.global_step % self.args.iteration_interval == 0:
            self.mcts_searcher.search_algo.policy_model = self.actor_reference_model if self.args.offline else self.actor_model
        target_probs, Q_values, r_values, base_values, visit_counts, select_indexes = [], [], [], [], [], []
        cur_node = None
        while cur_node is None or not cur_node.is_terminal:
            if cur_node is not None and (self.tokenizer.eos_token_id in cur_node.action or self.tokenizer.convert_tokens_to_ids("<|eot_id|>") in cur_node.action):
                cur_node.is_terminal = True
                break
            # MCTS for next step
            mcts_rst = self.mcts_searcher({
                'input_ids': seq, 'attention_mask': attn_msk,
                'answer': gt_answer, 'reasoning': solution,
                'answer_content': prompt_only_batch['answer_content'][0],
            }, node=cur_node)
            pi, cur_node = mcts_rst.next_action_pi, mcts_rst.tree_state
            target_probs.append(pi)
            Q_values.append([child.Q for child in cur_node.children])
            r_values.append([child.r for child in cur_node.children])
            base_values.append([child.value for child in cur_node.children])
            visit_counts.append([child.N for child in cur_node.children])
            
            cur_node = cur_node.children[mcts_rst.next_action_idx]
            select_indexes.append(mcts_rst.next_action_idx)
            
            if self.args.n_actions == 1: break
        
        dist.barrier()
        
        return [
            self.post_tree_construct(
                prompt=input_ids[idx],
                target_probs=target_probs,
                Q_values=Q_values,
                r_values=r_values,
                base_values=base_values,
                visit_counts=visit_counts,
                select_indexes=select_indexes,
                cur_node=mcts_rst.tree_state,
                solution=(solution, gt_answer,),
                cur_max_new_tokens=self.mcts_searcher.search_config.generation_config.max_new_tokens,
            )
            for idx in range(input_ids.size(0))
        ]
    
    def post_tree_construct(
        self,
        prompt: torch.Tensor,
        target_probs: list[torch.Tensor],
        Q_values: list[list[float]],
        r_values: list[list[float]],
        base_values: list[list[float]],
        visit_counts: list[list[int]],
        select_indexes: list[int],
        cur_node: MCTSNode,
        solution: tuple = None,
        cur_max_new_tokens: int = 32,
    ) -> dict[str, Any]:
        exec(f'''import pickle\nwith open('{self.args.output_dir}/mcts_rst.pkl', 'wb') as f: \n    pickle.dump(cur_node, f)''')
        
        while cur_node.depth:
            cur_node = cur_node.parent
        
        prompts, candidates, init_value_list, step_id = [], [], [], 0
        while cur_node.children:
            next_completions = []
            for child in cur_node.children:
                cur_child = child
                next_completion = [cur_child.action]
                while cur_child.children and len(cur_child.children) == 1:  # no other candidate(s)
                    cur_child = cur_child.children[0]
                    next_completion.append(cur_child.action)
                next_completions.append(torch.cat(next_completion, dim=-1))
            
            # record the scores: \pi (visiting count), Q values, advantages (relative values), base/init (absolute) values
            scores = [(q, s, r, bv, vc) for s, q, r, bv, vc in zip(target_probs[step_id], Q_values[step_id], r_values[step_id], base_values[step_id], visit_counts[step_id])]
            _candidates = [[x[1], scores[x[0]]] for x in sorted(enumerate(next_completions), key=lambda x: scores[x[0]])]
            init_values = [x[1][-1] for x in _candidates]   ## using visit count
            _candidates = [x[0] for x in _candidates]
            prompts.append(prompt)
            candidates.append(_candidates)
            init_value_list.append(init_values)
            
            cur_node = cur_node.children[select_indexes[step_id]]
            prompt = torch.cat([prompt, cur_node.action], dim=-1)
            step_id += 1
            while cur_node.children and len(cur_node.children) == 1:  # no other candidate(s)
                cur_node = cur_node.children[0]
                prompt = torch.cat([prompt, cur_node.action], dim=-1)
                step_id += 1
        
        mini_batches = {k:[] for k in ['prompts_list', 'input_ids_list', 'attention_mask_list', 'init_value_list']}
        for prompt, next_completions, init_values in zip(prompts, candidates, init_value_list):
            prompt = torch.stack([prompt for _ in next_completions], dim=0)
            attention_mask = pad_sequence([
                torch.ones((prompt.size(-1) + x.size(-1),), dtype=torch.bool, device=prompt.device)
                for x in next_completions
            ], batch_first=True, padding_value=False)
            next_completions = pad_sequence(next_completions, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            input_ids = torch.cat((prompt, next_completions), dim=-1)
            if input_ids.size(-1) > self.generation_config.max_length: continue
            mini_batches['prompts_list'].append(prompt)
            mini_batches['input_ids_list'].append(input_ids)
            mini_batches['attention_mask_list'].append(attention_mask)
            mini_batches['init_value_list'].append(init_values)
        
        if self.args.few_shot and self.args.model_type == 'gpt-j':
            qa_idx = get_final_qa_index(mini_batches['prompts_list'][0][0])
            mini_batches['prompts_list'] = [x[:, qa_idx:] for x in mini_batches['prompts_list']]
            mini_batches['input_ids_list'] = [x[:, qa_idx:] for x in mini_batches['input_ids_list']]
            mini_batches['attention_mask_list'] = [x[:, qa_idx:] for x in mini_batches['attention_mask_list']]
        
        r = max(r_values[-1])
        is_correct = False
        if len(mini_batches['input_ids_list']):
            text = self.tokenizer.decode(input_ids[-1], skip_special_tokens=True)
            if not text.startswith(PROMPT_BEGIN):
                prediction = text.split(PROMPT_ASSISTANT)[-1]
                if self.mcts_searcher.search_config.use_code:
                    is_correct = math_equal(extract_answer(prediction, use_code=self.mcts_searcher.search_config.use_code), solution[1])
                elif not solution[0].strip():
                    is_correct = csr_equal(prediction, ('(' + solution[1].strip() + ')', ''))
                else:
                    is_correct = math_equal(extract_answer(prediction), extract_answer(f'{solution[0]}\nThe answer is {solution[1]}'))
        
        mini_batches['prediction'] = (r, is_correct,)
        mini_batches['cur_max_new_tokens'] = cur_max_new_tokens
        return mini_batches

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
    
    def tsrl_step(
        self, 
        prompts_list: list[torch.Tensor], 
        input_ids_list: list[torch.Tensor],
        attention_mask_list: list[torch.Tensor],
        prediction: tuple = None,
        init_value_list: list[float] = None,
        max_n_sample: int = 8,
        cur_max_new_tokens: int = 32,
        num_step: int = 0,
    ) -> dict[str, Any]:
        losses, better_sample_rewards, worse_sample_rewards, max_lengths = [], [], [], []
        n_sample = len(input_ids_list)
        start = prompts_list[0].size(-1) - 1
        better_idx = -1
        worse_idx = 0 if self.args.choose_worst else -2
        
        all_better_input_ids, all_worse_input_ids = [], []
        all_better_attention_mask, all_worse_attention_mask = [], []
        all_init_value_list = []
        for sample_id in range(n_sample):
            if len(all_better_input_ids) >= max_n_sample: break
            
            input_ids = input_ids_list[sample_id]
            attention_mask = attention_mask_list[sample_id]
            
            n_output = input_ids.size(0)
            if n_output < 2: continue
            
            if self.args.choose_random:
                worse_idx = random.choice(range(n_output - 1))
                
            all_better_input_ids.append(input_ids[better_idx])
            all_worse_input_ids.append(input_ids[worse_idx])
            all_better_attention_mask.append(attention_mask[better_idx])
            all_worse_attention_mask.append(attention_mask[worse_idx])
            all_init_value_list.extend([init_value_list[sample_id][better_idx], init_value_list[sample_id][worse_idx]])
        all_input_ids = pad_tensors(all_better_input_ids + all_worse_input_ids, pad_value=self.tokenizer.pad_token_id)
        all_attention_mask = pad_tensors(all_better_attention_mask + all_worse_attention_mask, pad_value=False)
        
        torch.cuda.empty_cache()
        all_sequence_log_probs = self.compute_log_probs(
            self.actor_model.module,
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
        )
        all_better_input_ids, all_worse_input_ids = all_input_ids.chunk(chunks=2, dim=0)
        all_better_attention_mask, all_worse_attention_mask = all_attention_mask.chunk(chunks=2, dim=0)
        all_better_sequence_log_probs, all_worse_sequence_log_probs = all_sequence_log_probs.chunk(chunks=2, dim=0)
        
        label_smoothing_values = []
        for sample_id in range(len(all_better_input_ids)):
            better_input_ids = all_better_input_ids[sample_id]
            better_attention_mask = all_better_attention_mask[sample_id]
            
            worse_input_ids = all_worse_input_ids[sample_id]
            worse_attention_mask = all_worse_attention_mask[sample_id]
            
            init_values = [all_init_value_list[sample_id * 2], all_init_value_list[sample_id * 2 + 1]]
            better_sequence_log_probs, worse_sequence_log_probs = all_better_sequence_log_probs[sample_id], all_worse_sequence_log_probs[sample_id]
            
            with torch.no_grad():
                torch.cuda.empty_cache()
                ref_better_sequence_log_probs = self.compute_log_probs(
                    self.actor_reference_model.module,
                    input_ids=better_input_ids.unsqueeze(0),
                    attention_mask=better_attention_mask.unsqueeze(0),
                )[0]
                torch.cuda.empty_cache()
                ref_worse_sequence_log_probs = self.compute_log_probs(
                    self.actor_reference_model.module,
                    input_ids=worse_input_ids.unsqueeze(0),
                    attention_mask=worse_attention_mask.unsqueeze(0),
                )[0]
            
            better_end_index = better_attention_mask.nonzero()[-1]
            worse_end_index = worse_attention_mask.nonzero()[-1]
            try:
                diverge_index = (better_input_ids != worse_input_ids).nonzero()[0]
                assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
                assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'
            except:
                continue
            
            better_seq_slice = slice(diverge_index - 1, better_end_index)
            worse_seq_slice = slice(diverge_index - 1, worse_end_index)
            
            better_log_probs = better_sequence_log_probs[better_seq_slice].sum(dim=-1)
            worse_log_probs = worse_sequence_log_probs[worse_seq_slice].sum(dim=-1)
            ref_better_log_probs = ref_better_sequence_log_probs[better_seq_slice].sum(dim=-1)
            ref_worse_log_probs = ref_worse_sequence_log_probs[worse_seq_slice].sum(dim=-1)
            better_log_ratio = better_log_probs - ref_better_log_probs
            worse_log_ratio = worse_log_probs - ref_worse_log_probs
            if self.args.norm_prob or self.args.ipo:
                better_log_ratio /= better_attention_mask[better_seq_slice].sum(dim=-1) ** self.args.length_penalty
                worse_log_ratio /= worse_attention_mask[worse_seq_slice].sum(dim=-1) ** self.args.length_penalty
            logits = better_log_ratio - worse_log_ratio
            
            if self.args.ipo:
                losses.append((logits - 1 / (2 * self.scale_coeff)) ** 2)
            elif self.args.conservative:
                qb, qw = init_values
                confidence = calculate_preference_confidence(qb, qw)
                label_smoothing = min(1 - confidence, 0.5)
                losses.append(
                    - F.logsigmoid(self.scale_coeff * logits) * (1 - label_smoothing)
                    - F.logsigmoid(-self.scale_coeff * logits) * label_smoothing
                )
                label_smoothing_values.append(label_smoothing)
            else:
                losses.append(-F.logsigmoid(self.scale_coeff * logits))
            better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
            worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())
            
            max_lengths.append(better_attention_mask[start:].float().sum())
            max_lengths.append(worse_attention_mask[start:].float().sum())
        
        if not len(losses): return {}
        
        loss = torch.stack(losses).mean()
        max_generated_length = torch.stack(max_lengths).max()
        total_max_generated_length = max_generated_length + start
        better_sample_rewards = torch.stack(better_sample_rewards)  # size = (B,)
        worse_sample_rewards = torch.stack(worse_sample_rewards)  # size = (B,)
        rewards_accuracy = (
            (better_sample_rewards > worse_sample_rewards).float().mean()
        )  # size = ()
        better_sample_rewards = better_sample_rewards.mean()  # size = ()
        worse_sample_rewards = worse_sample_rewards.mean()  # size = ()
        rewards = better_sample_rewards + worse_sample_rewards  # size = ()
        rewards_margin = better_sample_rewards - worse_sample_rewards  # size = ()
        
        torch.cuda.empty_cache()
        self.actor_model.backward(loss)
        self.actor_model.step()
        
        loss = get_all_reduce_mean(loss)
        rewards = get_all_reduce_mean(rewards)
        better_sample_rewards = get_all_reduce_mean(better_sample_rewards)
        worse_sample_rewards = get_all_reduce_mean(worse_sample_rewards)
        rewards_accuracy = get_all_reduce_mean(rewards_accuracy)
        rewards_margin = get_all_reduce_mean(rewards_margin)
        max_generated_length = get_all_reduce_max(max_generated_length)
        total_max_generated_length = get_all_reduce_max(total_max_generated_length)
        
        return {
            'train/loss': loss.item(),
            'train/rewards': rewards.item(),
            'train/better_sample_rewards': better_sample_rewards.item(),
            'train/worse_sample_rewards': worse_sample_rewards.item(),
            'train/rewards_accuracy': rewards_accuracy.item(),
            'train/rewards_margin': rewards_margin.item(),
            'train/lr': self.actor_model.optimizer.param_groups[0]['lr'],
            'train/r_scores': float(prediction[0]),
            'train/correct': float(prediction[1]),
            'train/n_sample': n_sample,
            'train/max_generated_length': max_generated_length.item(),
            'train/total_max_generated_length': total_max_generated_length.item(),
            'train/label_smoothing': sum(label_smoothing_values) / len(label_smoothing_values) if len(label_smoothing_values) else 0,
            'train/cur_max_new_tokens': cur_max_new_tokens,
        }
    

class MCTSWDPOPTrainer(TSRLTrainer):
    TRAINING_TYPE = 'mcts'

    def init_mcts_searcher(self) -> None:
        world_model = StepLMWorldModel(
            max_length=self.generation_config.max_length,
            base_tokenizer=self.tokenizer,
            generation_config=self.generation_config,
        )
        search_cfg = StepLMConfig(SearchArgs(
            ref_policy_model=self.actor_reference_model,
            ref_policy_model2=self.actor_reference_model2,
            base_tokenizer=self.tokenizer,
            generation_config=self.generation_config,
            n_actions=self.args.n_actions,
            n_init_actions=self.args.n_init_actions,
            breadth_limit=self.args.breadth_limit,
            depth_limit=self.args.depth_limit,
            force_terminating_on_depth_limit=self.args.force_terminating_on_depth_limit,
            kl_coeff=self.args.kl_coeff,
            disable_tqdm=False,
            no_self_eval=self.args.no_self_eval,
            reward_model=self.reward_model if self.use_reward_model else None,
            reward_tokenizer=self.reward_tokenizer if self.use_reward_model else None,
            use_code=self.args.use_code,
            use_mcq=self.args.use_mcq,
            eval_mode=self.args.eval_mode,
            temperature=self.args.temperature,
            init_temperature=self.args.init_temperature,
            get_tp_zero=self.args.get_tp_zero,
            model_type=self.args.model_type,
            include_gt=(not self.args.not_include_gt),
            verbose=self.args.verbose,
        ))
        mcts_algo = MCTSWDPOP(MCTSConfig(
            w_exp=self.args.w_exp,
            depth_limit=self.args.depth_limit,
            breadth_limit=self.args.breadth_limit,
            n_iters=self.args.n_iters,
            temperature=self.args.mcts_temperature,
            temperature_decay_ratio=self.args.mcts_temperature_decay_ratio,
            consider_diversity=(not self.args.no_consider_diversity),
            length_penalty=self.args.mcts_length_penalty,
        ))
        self.mcts_searcher = TreeConstructor(
            world_model=world_model, 
            search_config=search_cfg, 
            search_algo=mcts_algo,
        )
    
    def tree_constructor(self, prompt_only_batch: PromptOnlyBatch | PromptOnlyPostBatch) -> list[dict[str, Any]]:
        """Rollout a batch of experiences."""
        input_ids = prompt_only_batch['input_ids']
        attention_mask = prompt_only_batch['attention_mask']
        answer = prompt_only_batch['answer']
        assert input_ids.size(0) == 1, '''Only support one instance per device.'''
        seq, attn_msk = input_ids[0], attention_mask[0]
        gt_answer, solution = answer[0], prompt_only_batch['reasoning'][0]
        
        # if solution.strip():
        #     self.mcts_searcher.search_config.generation_config.max_new_tokens = min(
        #         self.args.max_new_tokens,
        #         max(self.generation_config.max_new_tokens // 4,
        #             len(self.tokenizer.encode(solution)) // max(1, self.args.depth_limit - 1))
        #     )
        
        self.mcts_searcher.search_config.use_code = ('\nprint(' in solution)
        if self.mcts_searcher.search_algo.policy_model is None or self.global_step % self.args.iteration_interval == 0:
            self.mcts_searcher.search_algo.policy_model = self.actor_reference_model if self.args.offline else self.actor_model
        target_probs, Q_values, r_values, base_values, visit_counts, select_indexes = [], [], [], [], [], []
        cur_node = None
        while cur_node is None or not cur_node.is_terminal:
            if cur_node is not None and (self.tokenizer.eos_token_id in cur_node.action or self.tokenizer.convert_tokens_to_ids("<|eot_id|>") in cur_node.action):
                cur_node.is_terminal = True
                break
            # MCTS for next step
            mcts_rst = self.mcts_searcher({
                'input_ids': seq, 'attention_mask': attn_msk,
                'answer': gt_answer, 'reasoning': solution,
                'answer_content': prompt_only_batch['answer_content'][0],
            }, node=cur_node, policy_model=self.actor_model)
            pi, cur_node = mcts_rst.next_action_pi, mcts_rst.tree_state
            target_probs.append(pi)
            Q_values.append([child.Q for child in cur_node.children])
            r_values.append([child.r for child in cur_node.children])
            base_values.append([child.value for child in cur_node.children])
            visit_counts.append([child.N for child in cur_node.children])
            
            cur_node = cur_node.children[mcts_rst.next_action_idx]
            select_indexes.append(mcts_rst.next_action_idx)
            
            if self.args.n_actions == 1: break
        
        dist.barrier()
        
        return [
            self.post_tree_construct(
                prompt=input_ids[idx],
                target_probs=target_probs,
                Q_values=Q_values,
                r_values=r_values,
                base_values=base_values,
                visit_counts=visit_counts,
                select_indexes=select_indexes,
                cur_node=mcts_rst.tree_state,
                solution=(solution, gt_answer,),
                cur_max_new_tokens=self.mcts_searcher.search_config.generation_config.max_new_tokens,
            )
            for idx in range(input_ids.size(0))
        ]
    
    def gather_all_solutions(self, node, max_q, min_q):
        """
        Traverse the entire tree (rooted at `node`) and return a list of solutions.
        Each solution is a tuple: (step_actions, step_ws, sum_w).
        - step_actions: list of token Tensors from each node's .action
        - step_ws: list of normalized Q-values w_t for each node
        - sum_w: sum of all w_t in this path (useful for picking best/worst)
        """
        solutions = []

        def dfs(path_nodes):
            current_node = path_nodes[-1]
            if not current_node.children:
                # We reached a leaf => build a full solution from path_nodes
                step_actions, step_ws, ref_probs, sum_w = self.build_solution(path_nodes, max_q, min_q)
                solutions.append((step_actions, step_ws, ref_probs, sum_w))
                return

            for child in current_node.children:
                path_nodes.append(child)
                dfs(path_nodes)
                path_nodes.pop()

        # Start DFS from the root
        dfs([node])
        return solutions
    
    def build_solution(self, path_nodes, max_q, min_q):
        """
        Given a list of nodes [root, ..., leaf], build:
        - step_actions: [node_1.action, node_2.action, ...]
        - step_ws: [w_1, w_2, ...]  (each w_t is normalized from node.Q)
        - sum_w: sum of all w_t
        """
        step_actions = []
        step_ws = []
        ref_probs = []
        # If your root has no action, you can skip the root in iteration, or
        # include it if you prefer. This example starts from node_1 = path_nodes[1].
        for node in path_nodes[1:]:  
            # 1) The token(s) for this node
            step_actions.append(node.action)

            # 2) Convert node.Q -> w_t
            node_q = node.Q  # or wherever you store the Q
            if max_q > min_q:
                w_t = (node_q - min_q) / (max_q - min_q)
            else:
                w_t = 0.5
            step_ws.append(w_t)
            ref_probs.append(node.ref_log_probs.sum())

        sum_w = sum(step_ws)
        return step_actions, step_ws, ref_probs, sum_w
    
    def post_tree_construct(
        self,
        prompt: torch.Tensor,
        target_probs: list[torch.Tensor],
        Q_values: list[list[float]],
        r_values: list[list[float]],
        base_values: list[list[float]],
        visit_counts: list[list[int]],
        select_indexes: list[int],
        cur_node: MCTSNode,
        solution: tuple = None,
        cur_max_new_tokens: int = 32,
    ) -> dict[str, Any]:
        exec(f'''import pickle\nwith open('{self.args.output_dir}/mcts_rst.pkl', 'wb') as f: \n    pickle.dump(cur_node, f)''')
        
        while cur_node.depth:
            cur_node = cur_node.parent
        
        mini_batches = {
            # Full winner sequence (for DPO logistic difference)
            'prompts_list': [],
            'winner_input_ids_list': [],
            'winner_attention_mask_list': [],
            # Per-step actions & weights (for WDPOP hinge)
            'winner_step_actions_list': [],
            'winner_w_list': [],
            'winner_ref_probs': [],

            # Full loser sequence (for DPO logistic difference)
            'loser_input_ids_list': [],
            'loser_attention_mask_list': [],
            # Per-step actions & weights (for WDPOP hinge if desired)
            'loser_step_actions_list': [],
            'loser_w_list': [],
            'loser_ref_probs': [],
        }
        max_q = self.mcts_searcher.search_algo.max_q
        min_q = self.mcts_searcher.search_algo.min_q
        all_solutions = self.gather_all_solutions(cur_node, max_q, min_q)

        # 2) Identify winner (max sum_w) and loser (min sum_w)
        winner_idx = max(range(len(all_solutions)), key=lambda i: all_solutions[i][3])
        loser_idx  = min(range(len(all_solutions)), key=lambda i: all_solutions[i][3])


        winner_step_actions, winner_w_list, winner_ref_probs, _ = all_solutions[winner_idx]
        loser_step_actions,  loser_w_list,  loser_ref_probs, _ = all_solutions[loser_idx]

        winner_step_actions[0] = torch.cat([prompt, winner_step_actions[0]], dim=-1)
        loser_step_actions[0] = torch.cat([prompt, loser_step_actions[0]], dim=-1)

        # 3) Convert the entire winner path to one big input_ids
        winner_input_ids = torch.cat(winner_step_actions, dim=-1)
        winner_attention_mask = torch.ones(
            winner_input_ids.size(-1),
            dtype=torch.bool,
            device=winner_input_ids.device
        )

        # 4) Convert the entire loser path to one big input_ids
        loser_input_ids = torch.cat(loser_step_actions, dim=-1)
        loser_attention_mask = torch.ones(
            loser_input_ids.size(-1),
            dtype=torch.bool,
            device=loser_input_ids.device
        )

        # 5) Store them in mini_batches
        #    For the DPO logistic difference
        mini_batches['prompts_list'].append(prompt)
        mini_batches['winner_input_ids_list'].append(winner_input_ids.unsqueeze(0))
        mini_batches['winner_attention_mask_list'].append(winner_attention_mask.unsqueeze(0))
        mini_batches['loser_input_ids_list'].append(loser_input_ids.unsqueeze(0))
        mini_batches['loser_attention_mask_list'].append(loser_attention_mask.unsqueeze(0))

        #    For the WDPOP hinge, we also store the "step_actions" and "w_list"
        mini_batches['winner_step_actions_list'].append(winner_step_actions)  # list of Tensors
        mini_batches['winner_w_list'].append(winner_w_list)                  # list of floats
        mini_batches['loser_step_actions_list'].append(loser_step_actions)
        mini_batches['loser_w_list'].append(loser_w_list)
        mini_batches['winner_ref_probs'].append(winner_ref_probs)
        mini_batches['loser_ref_probs'].append(loser_ref_probs)

        # (Optional) correctness check
        r = max(r_values[-1]) if r_values else 0
        is_correct = False
        if len(mini_batches['winner_input_ids_list']):
            text = self.tokenizer.decode(winner_input_ids, skip_special_tokens=True)
            if not text.startswith(PROMPT_BEGIN):
                prediction = text.split(PROMPT_ASSISTANT)[-1]
                if self.mcts_searcher.search_config.use_code:
                    is_correct = math_equal(extract_answer(prediction, use_code=self.mcts_searcher.search_config.use_code), solution[1])
                elif not solution[0].strip():
                    is_correct = csr_equal(prediction, ('(' + solution[1].strip() + ')', ''))
                else:
                    is_correct = math_equal(extract_answer(prediction), extract_answer(f'{solution[0]}\nThe answer is {solution[1]}'))
        
        mini_batches['prediction'] = (r, is_correct,)
        mini_batches['cur_max_new_tokens'] = cur_max_new_tokens

        # self.replay_buffer.add(mini_batches)
        return mini_batches

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        
    def pad_tensors(self, tensors: list[torch.Tensor], max_len: int = -1, pad_value: int = IGNORE_INDEX) -> torch.Tensor:
        processed = []
        for tensor in tensors:
            # If tensor shape is [1, seq_len], squeeze to [seq_len]
            if tensor.dim() > 1 and tensor.size(0) == 1:
                processed.append(tensor.squeeze(0))
            else:
                processed.append(tensor)
        if max_len <= 0:
            max_len = max(tensor.size(-1) for tensor in processed)
        padded_tensors = []
        for tensor in processed:
            cur_len = tensor.size(-1)
            pad_len = max_len - cur_len
            if pad_len > 0:
                pad_shape = list(tensor.shape)
                pad_shape[-1] = pad_len
                pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
                padded_tensor = torch.cat([tensor, pad_tensor], dim=-1)
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)
        return torch.stack(padded_tensors, dim=0)
    def tsrl_step(
        self,
        # Winner tensors: each is a Tensor (e.g., [1, seq_len]) or list of per-step Tensors.
        winner_input_ids_list: list[torch.Tensor],
        winner_attention_mask_list: list[torch.Tensor],
        winner_step_actions_list: list[list[torch.Tensor]],
        winner_w_list: list[list[float]],
        winner_ref_probs: list[list[float]],
        prompts_list: list[torch.Tensor],

        # Loser tensors.
        loser_input_ids_list: list[torch.Tensor],
        loser_attention_mask_list: list[torch.Tensor],
        loser_step_actions_list: list[list[torch.Tensor]],
        loser_w_list: list[list[float]],
        loser_ref_probs: list[list[float]],

        prediction: tuple = (0.0, False),
        cur_max_new_tokens: int = 32,
        beta: float = 1.0,         # WDPOP coefficient on log-ratios
        lambda_: float = 1.0,      # WDPOP hinge weight
        num_step: int = 0,
    ) -> dict[str, Any]:
        """
        Implements the WDPOP loss:

        L_{WDPOP}(θ) = - E_{(τ^w, τ^l)} [
            log σ(
                beta * (sum_t (log π_θ - log π_ref)_winner
                    - sum_t (log π_θ - log π_ref)_loser)
                + lambda * (sum_t beta * w_t * max(0, log π_ref - log π_θ)_winner)
            )
        ]
        
        This version batches winner and loser sequences together by padding them to the same length.
        Note: Using the same model for policy and reference will yield diff = 0.
        """
        device = winner_input_ids_list[0].device if winner_input_ids_list else "cpu"

        # -----------------------------------
        # Combine and pad winner and loser sequences.
        # -----------------------------------
        all_winner_input_ids = winner_input_ids_list
        all_winner_attention_mask = winner_attention_mask_list
        all_loser_input_ids = loser_input_ids_list
        all_loser_attention_mask = loser_attention_mask_list

        if not all_winner_input_ids:
            return {}

        # Concatenate winner and loser lists.
        combined_input_ids = self.pad_tensors(all_winner_input_ids + all_loser_input_ids,
                                        pad_value=self.tokenizer.pad_token_id)
        combined_attention_mask = self.pad_tensors(all_winner_attention_mask + all_loser_attention_mask,
                                            pad_value=False)

        # Determine the number of winner sequences.
        n_winner = len(all_winner_input_ids)

        # Split the combined padded tensors back into winner and loser batches.
        winner_input_ids_batch = combined_input_ids[:n_winner]
        loser_input_ids_batch = combined_input_ids[n_winner:]
        winner_attention_mask_batch = combined_attention_mask[:n_winner]
        loser_attention_mask_batch = combined_attention_mask[n_winner:]

        # -----------------------------------
        # Actor model forward pass: process both batches at once.
        # -----------------------------------
        torch.cuda.empty_cache()
        combined_log_probs = self.compute_log_probs(
            self.actor_model.module,
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask,
        )
        # torch.cuda.empty_cache()
        # winner_log_probs_batch = self.compute_log_probs(
        #     self.actor_model.module,
        #     input_ids=winner_input_ids_batch,
        #     attention_mask=winner_attention_mask_batch,
        # )
        # torch.cuda.empty_cache()
        # loser_log_probs_batch = self.compute_log_probs(
        #     self.actor_model.module,
        #     input_ids=loser_input_ids_batch,
        #     attention_mask=loser_attention_mask_batch,
        # )
        winner_log_probs_batch = combined_log_probs[:n_winner]
        loser_log_probs_batch = combined_log_probs[n_winner:]

        # -----------------------------------
        # Reference model forward passes (separate).
        # -----------------------------------
        with torch.no_grad():
            torch.cuda.empty_cache()
            ref_winner_log_probs_batch = self.compute_log_probs(
                self.actor_reference_model.module,
                input_ids=winner_input_ids_batch,
                attention_mask=winner_attention_mask_batch,
            )
            torch.cuda.empty_cache()
            ref_loser_log_probs_batch = self.compute_log_probs(
                self.actor_reference_model.module,
                input_ids=loser_input_ids_batch,
                attention_mask=loser_attention_mask_batch,
            )

        # -----------------------------------
        # Accumulate WDPOP loss per sample.
        # -----------------------------------
        losses = []
        hinge_terms = []
        for i in range(len(winner_input_ids_list)):
            # Determine actual lengths (excluding padding) minus one token.
            w_len = int(winner_attention_mask_list[i].sum().item()) - 1
            l_len = int(loser_attention_mask_list[i].sum().item()) - 1

            # Slice the log-probabilities.
            w_log_probs = winner_log_probs_batch[i][:w_len]
            l_log_probs = loser_log_probs_batch[i][:l_len]
            ref_w_log_probs = ref_winner_log_probs_batch[i][:w_len]
            ref_l_log_probs = ref_loser_log_probs_batch[i][:l_len]

            # Compute the overall log-ratio difference.
            sum_winner_ratio = beta * (w_log_probs.sum() - ref_w_log_probs.sum())
            sum_loser_ratio  = beta * (l_log_probs.sum() - ref_l_log_probs.sum())


            sum_winner_ratio = beta * w_log_probs.sum() 
            sum_loser_ratio  = beta * l_log_probs.sum() 
            # if num_step == 0:
            #     sum_winner_ratio = beta * w_log_probs.sum()
            #     sum_loser_ratio  = beta * l_log_probs.sum()
            
            diff = sum_winner_ratio - sum_loser_ratio

            # Compute the hinge penalty for the winner trajectory.
            step_hinge = 0.0
            step_actions = winner_step_actions_list[i]  # List of per-step Tensors.
            step_weights = winner_w_list[i]             # List of corresponding weights.
            running_idx = 0
            for actions_t, w_t in zip(step_actions, step_weights):
                step_len = actions_t.size(-1)
                step_log_theta = w_log_probs[running_idx: running_idx + step_len].sum()
                step_log_ref   = ref_w_log_probs[running_idx: running_idx + step_len].sum()
                running_idx += step_len
                # Hinge: beta * w_t * max(0, (log π_ref - log π_θ))
                step_hinge += beta * w_t * torch.clamp(step_log_ref - step_log_theta, min=0.0)
                # if num_step == 0:
                #     step_hinge += beta * w_t * (-step_log_theta)
            hinge_penalty = lambda_ * step_hinge

            # Compute the WDPOP loss for this sample.
            total_loss_i = -F.logsigmoid(diff - hinge_penalty)
            losses.append(total_loss_i)
            hinge_terms.append(step_hinge.item())

        if not losses:
            return {}

        loss = torch.stack(losses).mean()
        # torch.cuda.empty_cache()
        # Backward pass and optimizer step.
        self.actor_model.backward(loss)
        self.actor_model.step()
        total_grad_norm = 0.0
        for param in self.actor_model.module.parameters():
            if param.grad is not None:
                print('grad is not none')
                print(param.grad)
                total_grad_norm += (param.grad.data**2).sum().item()
        print("Grad norm:", total_grad_norm**0.5)
        # Gather statistics.
        loss_val = get_all_reduce_mean(loss).item()
        hinge_val = get_all_reduce_mean(torch.tensor(hinge_terms, device=device)).item()
        r, is_correct = prediction

        return {
            "train/loss": loss_val,
            "train/hinge": hinge_val,
            "train/r_scores": float(r),
            "train/correct": float(is_correct),
            "train/n_sample": len(winner_input_ids_list),
            "train/cur_max_new_tokens": cur_max_new_tokens,
        }
    # def tsrl_step(
    #     self,
    #     # Winner tensors: each is a Tensor (e.g., [1, seq_len]) or list of per-step Tensors.
    #     winner_input_ids_list: list[torch.Tensor],
    #     winner_attention_mask_list: list[torch.Tensor],
    #     winner_step_actions_list: list[list[torch.Tensor]],
    #     winner_w_list: list[list[float]],
    #     prompts_list: list[torch.Tensor],

    #     # Loser tensors.
    #     loser_input_ids_list: list[torch.Tensor],
    #     loser_attention_mask_list: list[torch.Tensor],
    #     loser_step_actions_list: list[list[torch.Tensor]],
    #     loser_w_list: list[list[float]],

    #     prediction: tuple = (0.0, False),
    #     cur_max_new_tokens: int = 32,
    #     beta: float = 1.0,         # WDPOP coefficient on log-ratios
    #     lambda_: float = 1.0,      # WDPOP hinge weight
    #     num_step: int = 0,
    # ) -> dict[str, Any]:
    #     """
    #     Implements the WDPOP loss:

    #     L_{WDPOP}(θ) = - E_{(τ^w, τ^l)} [
    #         log σ(
    #             beta * (sum_t (log π_θ - log π_ref)_winner
    #                 - sum_t (log π_θ - log π_ref)_loser)
    #             + lambda * (sum_t beta * w_t * max(0, log π_ref - log π_θ)_winner)
    #         )
    #     ]
        
    #     This version batches winner and loser sequences together by padding them to the same length.
    #     Note: Using the same model for policy and reference will yield diff = 0.
    #     """
    #     device = winner_input_ids_list[0].device if winner_input_ids_list else "cpu"

    #     # -----------------------------------
    #     # Combine and pad winner and loser sequences.
    #     # -----------------------------------
    #     all_winner_input_ids = winner_input_ids_list
    #     all_winner_attention_mask = winner_attention_mask_list
    #     all_loser_input_ids = loser_input_ids_list
    #     all_loser_attention_mask = loser_attention_mask_list

    #     if not all_winner_input_ids:
    #         return {}

    #     # Concatenate winner and loser lists.
    #     combined_input_ids = self.pad_tensors(all_winner_input_ids + all_loser_input_ids,
    #                                     pad_value=self.tokenizer.pad_token_id)
    #     combined_attention_mask = self.pad_tensors(all_winner_attention_mask + all_loser_attention_mask,
    #                                         pad_value=False)

    #     # Determine the number of winner sequences.
    #     n_winner = len(all_winner_input_ids)

    #     # Split the combined padded tensors back into winner and loser batches.
    #     winner_input_ids_batch = combined_input_ids[:n_winner]
    #     loser_input_ids_batch = combined_input_ids[n_winner:]
    #     winner_attention_mask_batch = combined_attention_mask[:n_winner]
    #     loser_attention_mask_batch = combined_attention_mask[n_winner:]

    #     # -----------------------------------
    #     # Actor model forward pass: process both batches at once.
    #     # -----------------------------------
    #     torch.cuda.empty_cache()
    #     combined_log_probs = self.compute_log_probs(
    #         self.actor_model.module,
    #         input_ids=combined_input_ids,
    #         attention_mask=combined_attention_mask,
    #     )
    #     # Split outputs back.
    #     winner_log_probs_batch = combined_log_probs[:n_winner]
    #     loser_log_probs_batch = combined_log_probs[n_winner:]

    #     # -----------------------------------
    #     # Reference model forward passes (separate).
    #     # -----------------------------------
    #     with torch.no_grad():
    #         torch.cuda.empty_cache()
    #         ref_winner_log_probs_batch = self.compute_log_probs(
    #             self.actor_reference_model.module,
    #             input_ids=winner_input_ids_batch,
    #             attention_mask=winner_attention_mask_batch,
    #         )
    #         torch.cuda.empty_cache()
    #         ref_loser_log_probs_batch = self.compute_log_probs(
    #             self.actor_reference_model.module,
    #             input_ids=loser_input_ids_batch,
    #             attention_mask=loser_attention_mask_batch,
    #         )

    #     # -----------------------------------
    #     # Accumulate WDPOP loss per sample.
    #     # -----------------------------------
    #     losses = []
    #     hinge_terms = []
    #     for i in range(len(winner_input_ids_list)):
    #         # Determine actual lengths (excluding padding) minus one token.
    #         w_len = int(winner_attention_mask_list[i].sum().item()) - 1
    #         l_len = int(loser_attention_mask_list[i].sum().item()) - 1

    #         # Slice the log-probabilities.
    #         w_log_probs = winner_log_probs_batch[i][:w_len]
    #         l_log_probs = loser_log_probs_batch[i][:l_len]
    #         ref_w_log_probs = ref_winner_log_probs_batch[i][:w_len]
    #         ref_l_log_probs = ref_loser_log_probs_batch[i][:l_len]

    #         # Compute the overall log-ratio difference.
    #         sum_winner_ratio = beta * (w_log_probs.sum() - ref_w_log_probs.sum())
    #         sum_loser_ratio  = beta * (l_log_probs.sum() - ref_l_log_probs.sum())
    #         # if num_step == 0:
    #         #     sum_winner_ratio = beta * w_log_probs.sum()
    #         #     sum_loser_ratio  = beta * l_log_probs.sum()
            
    #         diff = sum_winner_ratio - sum_loser_ratio

    #         # Compute the hinge penalty for the winner trajectory.
    #         step_hinge = 0.0
    #         step_actions = winner_step_actions_list[i]  # List of per-step Tensors.
    #         step_weights = winner_w_list[i]             # List of corresponding weights.
    #         running_idx = 0
    #         for actions_t, w_t in zip(step_actions, step_weights):
    #             step_len = actions_t.size(-1)
    #             step_log_theta = w_log_probs[running_idx: running_idx + step_len].sum()
    #             step_log_ref   = ref_w_log_probs[running_idx: running_idx + step_len].sum()
    #             running_idx += step_len
    #             # Hinge: beta * w_t * max(0, (log π_ref - log π_θ))
    #             step_hinge += beta * w_t * torch.clamp(step_log_ref - step_log_theta, min=0.0)
    #             # if num_step == 0:
    #             #     step_hinge += beta * w_t * (-step_log_theta)
    #         hinge_penalty = lambda_ * step_hinge

    #         # Compute the WDPOP loss for this sample.
    #         total_loss_i = -F.logsigmoid(diff - hinge_penalty)
    #         losses.append(total_loss_i)
    #         hinge_terms.append(step_hinge.item())

    #     if not losses:
    #         return {}

    #     loss = torch.stack(losses).mean()
    #     # torch.cuda.empty_cache()
    #     # Backward pass and optimizer step.
    #     self.actor_model.backward(loss)
    #     self.actor_model.step()

    #     # Gather statistics.
    #     loss_val = get_all_reduce_mean(loss).item()
    #     hinge_val = get_all_reduce_mean(torch.tensor(hinge_terms, device=device)).item()
    #     r, is_correct = prediction

    #     return {
    #         "train/loss": loss_val,
    #         "train/hinge": hinge_val,
    #         "train/r_scores": float(r),
    #         "train/correct": float(is_correct),
    #         "train/n_sample": len(winner_input_ids_list),
    #         "train/cur_max_new_tokens": cur_max_new_tokens,
    #     }

    
    # def tsrl_step(
    #     self,
    #     # Each of these is a list of length B (batch size).
    #     # Inside, each element is a Tensor of shape [seq_len] or a list of Tensors (for step_actions).
    #     winner_input_ids_list: list[torch.Tensor],
    #     winner_attention_mask_list: list[torch.Tensor],
    #     winner_step_actions_list: list[list[torch.Tensor]],
    #     winner_w_list: list[list[float]],
    #     prompts_list: list[torch.Tensor], 

    #     loser_input_ids_list: list[torch.Tensor],
    #     loser_attention_mask_list: list[torch.Tensor],
    #     loser_step_actions_list: list[list[torch.Tensor]],
    #     loser_w_list: list[list[float]],

    #     prediction: tuple = (0.0, False),
    #     cur_max_new_tokens: int = 32,
    #     beta: float = 1.0,         # WDPOP coefficient on log-ratios
    #     lambda_: float = 1.0,      # WDPOP hinge weight
    #     ) -> dict[str, Any]:
    #     """
    #     Implements the WDPOP loss:

    #         L_{WDPOP}(θ) = - E_{(τ^w, τ^l)} [
    #         log σ( Sum_t (beta * log π_θ/π_ref) (winner) - Sum_t (beta * log π_θ/π_ref) (loser) 
    #         - lambda * Sum_t (beta * w_t * max(0, log π_ref/π_θ (winner))) )
    #         ]

    #     where (τ^w, τ^l) are the winner and loser trajectories.
    #     """


    #     device = winner_input_ids_list[0].device if len(winner_input_ids_list) else 'cpu'

    #     # Collect all winner & loser sequences in a single batch so we can do a single forward pass
    #     # for efficiency. Then we'll chunk the results back out.
    #     all_winner_input_ids = []
    #     all_winner_attention_mask = []
    #     all_loser_input_ids = []
    #     all_loser_attention_mask = []

    #     for w_inp, w_attn, l_inp, l_attn in zip(
    #         winner_input_ids_list,
    #         winner_attention_mask_list,
    #         loser_input_ids_list,
    #         loser_attention_mask_list,
    #     ):
    #         all_winner_input_ids.append(w_inp)
    #         all_winner_attention_mask.append(w_attn)
    #         all_loser_input_ids.append(l_inp)
    #         all_loser_attention_mask.append(l_attn)

    #     # Pad all winner sequences together, and all loser sequences together
    #     # (You already have pad_tensors in your code.)
    #     if len(all_winner_input_ids) == 0:
    #         return {}

    #     winner_input_ids_batch = pad_tensors(all_winner_input_ids, pad_value=self.tokenizer.pad_token_id)
    #     winner_attention_mask_batch = pad_tensors(all_winner_attention_mask, pad_value=False)
    #     loser_input_ids_batch = pad_tensors(all_loser_input_ids, pad_value=self.tokenizer.pad_token_id)
    #     loser_attention_mask_batch = pad_tensors(all_loser_attention_mask, pad_value=False)

    #     # 1) Compute log-probs under π_θ
    #     # with torch.no_grad():
    #         # If your actor_model is wrapped in DataParallel/Distributed, adapt accordingly
    #         # Winner
    #     winner_input_ids_batch = winner_input_ids_batch.squeeze(0)
    #     winner_attention_mask_batch = winner_attention_mask_batch.squeeze(0)
    #     loser_input_ids_batch = loser_input_ids_batch.squeeze(0)
    #     loser_attention_mask_batch = loser_attention_mask_batch.squeeze(0)
    #     torch.cuda.empty_cache()
    #     winner_log_probs_batch = self.compute_log_probs(
    #         self.actor_model.module,
    #         winner_input_ids_batch,
    #         winner_attention_mask_batch
    #     )
    #     # Loser
    #     torch.cuda.empty_cache()
    #     loser_log_probs_batch = self.compute_log_probs(
    #         self.actor_model.module,
    #         loser_input_ids_batch,
    #         loser_attention_mask_batch
    #     )

    #     # 2) Compute log-probs under π_ref
    #     with torch.no_grad():
    #         torch.cuda.empty_cache()
    #         ref_winner_log_probs_batch = self.compute_log_probs(
    #             self.actor_reference_model.module,
    #             winner_input_ids_batch,
    #             winner_attention_mask_batch
    #         )
    #         torch.cuda.empty_cache()
    #         ref_loser_log_probs_batch = self.compute_log_probs(
    #             self.actor_reference_model.module,
    #             loser_input_ids_batch,
    #             loser_attention_mask_batch
    #         )

    #     # Now accumulate the WDPOP loss across the batch
    #     losses = []
    #     hinge_terms = []
    #     for i in range(len(winner_input_ids_list)):
    #         # Extract the unpadded piece from the big batch
    #         # shape: [seq_len_i-1] for log_probs
    #         w_len = winner_attention_mask_list[i].sum().long().item() - 1
    #         l_len = loser_attention_mask_list[i].sum().long().item() - 1

    #         w_log_probs = winner_log_probs_batch[i][:w_len]
    #         l_log_probs = loser_log_probs_batch[i][:l_len]
    #         ref_w_log_probs = ref_winner_log_probs_batch[i][:w_len]
    #         ref_l_log_probs = ref_loser_log_probs_batch[i][:l_len]

    #         # 2a) Sum of log-ratios for winner vs. loser
    #         #     ratio = sum of (log π_θ - log π_ref)
    #         #     multiply by beta
    #         sum_winner_ratio = beta * (w_log_probs.sum() - ref_w_log_probs.sum())
    #         sum_loser_ratio  = beta * (l_log_probs.sum() - ref_l_log_probs.sum())
    #         diff = sum_winner_ratio - sum_loser_ratio  # Δ in the formula

    #         # 2b) The logistic difference part:  - log σ(diff)
    #         logistic_term = -F.logsigmoid(diff)

    #         # 2c) The hinge penalty:  - lambda * sum_{t=0}^{N-1} [ beta * w_t * max(0, log π_ref / π_θ ) ]
    #         #     We do it "per-step" using winner_step_actions_list[i] and winner_w_list[i].
    #         #     For each step t, we find the log-ratio over that step's tokens.
    #         step_hinge = 0.0
    #         step_actions = winner_step_actions_list[i]  # list of Tensors
    #         step_weights = winner_w_list[i]             # list of floats

    #         # We'll keep a running index in w_log_probs for each "step."
    #         # If each step is just 1 token, this is straightforward.
    #         # If each step has multiple tokens, sum them. Adjust accordingly.
    #         running_idx = 0
    #         for t, (actions_t, w_t) in enumerate(zip(step_actions, step_weights)):
    #             step_len = actions_t.size(-1)
    #             # sum of log(π_θ) - sum of log(π_ref) for the tokens in step t
    #             step_log_theta = w_log_probs[running_idx : running_idx + step_len].sum()
    #             step_log_ref   = ref_w_log_probs[running_idx : running_idx + step_len].sum()
    #             running_idx += step_len

    #             # hinge = beta * w_t * max(0, (step_log_ref - step_log_theta))
    #             # i.e. max(0, log π_ref/π_θ ) = max(0, step_log_ref - step_log_theta)
    #             step_hinge_val = beta * w_t * torch.clamp(step_log_ref - step_log_theta, min=0.0)
    #             step_hinge += step_hinge_val

    #         hinge_penalty = lambda_ * step_hinge
    #         total_loss_i = -F.logsigmoid(diff + hinge_penalty)  # matches the sign in your WDPOP formula
    #         losses.append(total_loss_i)

    #         hinge_terms.append(step_hinge.item())

    #     if not losses:
    #         return {}

    #     # Combine the losses over the batch
    #     loss = torch.stack(losses).mean()
    #     torch.cuda.empty_cache()
    #     # Backprop
    #     self.actor_model.backward(loss)
    #     self.actor_model.step()

    #     # Optional: gather stats
    #     loss_val = get_all_reduce_mean(loss).item()
    #     hinge_val = get_all_reduce_mean(torch.tensor(hinge_terms, device=device)).item()

    #     # You can still measure "accuracy" if you want, though WDPOP's objective is about preference
    #     # We'll just replicate some of your logging:
    #     r, is_correct = prediction

    #     return {
    #         "train/loss": loss_val,
    #         "train/hinge": hinge_val,
    #         "train/r_scores": float(r),
    #         "train/correct": float(is_correct),
    #         "train/n_sample": len(winner_input_ids_list),
    #         "train/cur_max_new_tokens": cur_max_new_tokens,
    #         # etc. Add any other stats you want
    #     }

    # def tsrl_step(
    #     self,
    #     winner_input_ids_list: list[torch.Tensor],
    #     winner_attention_mask_list: list[torch.Tensor],
    #     winner_step_actions_list: list[list[torch.Tensor]],
    #     winner_w_list: list[list[float]],

    #     loser_input_ids_list: list[torch.Tensor],
    #     loser_attention_mask_list: list[torch.Tensor],
    #     loser_step_actions_list: list[list[torch.Tensor]],
    #     loser_w_list: list[list[float]],

    #     prediction: tuple = None,
    #     max_n_sample: int = 8,
    #     cur_max_new_tokens: int = 32,
    # ) -> dict[str, Any]:
    #     losses = []
    #     better_sample_rewards, worse_sample_rewards = [], []

    #     n_sample = len(winner_input_ids_list)
    #     for sample_id in range(n_sample):
    #         if sample_id >= max_n_sample:
    #             break

    #         # 1) Entire path for winner vs. loser (DPO logistic)
    #         winner_input_ids = winner_input_ids_list[sample_id][0]  # shape [seq_len_w]
    #         winner_mask      = winner_attention_mask_list[sample_id][0]

    #         loser_input_ids  = loser_input_ids_list[sample_id][0]   # shape [seq_len_l]
    #         loser_mask       = loser_attention_mask_list[sample_id][0]
    #         torch.cuda.empty_cache()
    #         winner_seq_log_probs = self.compute_log_probs(
    #             self.actor_model.module,
    #             winner_input_ids.unsqueeze(0),
    #             winner_mask.unsqueeze(0),
    #         )[0].sum()
    #         #@TODO: with gradient calculation for policy model, not ref model
    #         with torch.no_grad():
    #             torch.cuda.empty_cache()
    #             ref_winner_seq_log_probs = self.compute_log_probs(
    #                 self.actor_reference_model.module,
    #                 winner_input_ids.unsqueeze(0),
    #                 winner_mask.unsqueeze(0),
    #             )[0].sum()

    #             loser_seq_log_probs = self.compute_log_probs(
    #                 self.actor_model.module,
    #                 loser_input_ids.unsqueeze(0),
    #                 loser_mask.unsqueeze(0),
    #             )[0].sum()
    #             ref_loser_seq_log_probs = self.compute_log_probs(
    #                 self.actor_reference_model.module,
    #                 loser_input_ids.unsqueeze(0),
    #                 loser_mask.unsqueeze(0),
    #             )[0].sum()

    #         # DPO difference
    #         winner_log_ratio = winner_seq_log_probs - ref_winner_seq_log_probs
    #         loser_log_ratio  = loser_seq_log_probs  - ref_loser_seq_log_probs
    #         logits = winner_log_ratio - loser_log_ratio

    #         dpo_loss = -F.logsigmoid(self.scale_coeff * logits)

    #         # 2) Stepwise WDPOP hinge
    #         #    For each step t in the winner, we get a corresponding step t in the loser
    #         #    Then sum up w_t * ReLU( log π^\nu(a_t^\nu) - log π^\mu(a_t^\mu) ).
    #         #    If the # steps differ, we do a partial match or skip extra steps.
    #         winner_steps = winner_step_actions_list[sample_id]  # list of Tensors
    #         loser_steps  = loser_step_actions_list[sample_id]
    #         w_list_winner = winner_w_list[sample_id]
            
    #         n_steps = len(winner_steps)
    #         hinge_sum = 0.0

    #         for t in range(n_steps):
    #             w_t = w_list_winner[t]  # weight from the winner
    #             winner_action = winner_steps[t]  # shape [k1]
    #             loser_action  = loser_steps[t]    # shape [k2] 
    #             # If you want them to be the same shape, your tree expansions must align. 
    #             # We'll do a forward pass on each chunk to get log π^\nu, log π^\mu.
    #             #@TODO: ref model 
    #             with torch.no_grad():
    #                 # compute log probs for this step chunk
    #                 # We'll treat each step as a separate sequence (be sure to add context if needed).
    #                 # For simplicity, assume each step stands alone. 
    #                 # If you need context from previous steps, you must cat them cumulatively.
    #                 winner_step_log_prob = self.compute_log_probs(
    #                     self.actor_model.module,
    #                     winner_action.unsqueeze(0),
    #                     torch.ones_like(winner_action).unsqueeze(0),  # simple attention mask
    #                 )[0].sum()

    #                 loser_step_log_prob = self.compute_log_probs(
    #                     self.actor_model.module,
    #                     loser_action.unsqueeze(0),
    #                     torch.ones_like(loser_action).unsqueeze(0),
    #                 )[0].sum()

    #             step_diff = winner_step_log_prob - loser_step_log_prob
    #             hinge_sum += w_t * F.relu(step_diff)

    #         # Multiply by -λ to add to the objective
    #         hinge_term = - self.args.lambda_ * hinge_sum

    #         total_loss = dpo_loss + hinge_term
    #         losses.append(total_loss)

    #         # For logging
    #         better_sample_rewards.append(winner_log_ratio.detach())
    #         worse_sample_rewards.append(loser_log_ratio.detach())

    #     # 3) Final optimization
    #     if not losses:
    #         return {}

    #     loss = torch.stack(losses).mean()
    #     self.actor_model.backward(loss)
    #     self.actor_model.step()

    #     # 4) Logging
    #     better_sample_rewards = torch.stack(better_sample_rewards)
    #     worse_sample_rewards  = torch.stack(worse_sample_rewards)
    #     rewards_accuracy = (better_sample_rewards > worse_sample_rewards).float().mean()

    #     return {
    #         'train/loss': loss.item(),
    #         'train/rewards_accuracy': rewards_accuracy.item(),
    #         'train/r_scores': float(prediction[0]) if prediction else 0,
    #         'train/correct': float(prediction[1]) if prediction else 0,
    #     }
        # losses, better_sample_rewards, worse_sample_rewards, max_lengths = [], [], [], []
        # n_sample = len(winner_input_ids_list)
        # start = prompts_list[0].size(-1) - 1
        # better_idx = -1
        # worse_idx = 0 if self.args.choose_worst else -2
        
        # all_better_input_ids, all_worse_input_ids = [], []
        # all_better_attention_mask, all_worse_attention_mask = [], []
        # all_init_value_list = []
        # for sample_id in range(n_sample):
        #     if len(all_better_input_ids) >= max_n_sample: break
        #     # 1) Entire path for winner vs. loser (DPO logistic)
        #     winner_input_ids = winner_input_ids_list[sample_id][0]  # shape [seq_len_w]
        #     winner_mask      = winner_attention_mask_list[sample_id][0]

        #     loser_input_ids  = loser_input_ids_list[sample_id][0]   # shape [seq_len_l]
        #     loser_mask       = loser_attention_mask_list[sample_id][0]

        #     input_ids = input_ids_list[sample_id]
        #     attention_mask = attention_mask_list[sample_id]
            
        #     n_output = input_ids.size(0)
        #     if n_output < 2: continue
            
        #     if self.args.choose_random:
        #         worse_idx = random.choice(range(n_output - 1))
                
        #     all_better_input_ids.append(input_ids[better_idx])
        #     all_worse_input_ids.append(input_ids[worse_idx])
        #     all_better_attention_mask.append(attention_mask[better_idx])
        #     all_worse_attention_mask.append(attention_mask[worse_idx])
        #     all_init_value_list.extend([init_value_list[sample_id][better_idx], init_value_list[sample_id][worse_idx]])
        # all_input_ids = pad_tensors(all_better_input_ids + all_worse_input_ids, pad_value=self.tokenizer.pad_token_id)
        # all_attention_mask = pad_tensors(all_better_attention_mask + all_worse_attention_mask, pad_value=False)
        
        # torch.cuda.empty_cache()
        # all_sequence_log_probs = self.compute_log_probs(
        #     self.actor_model.module,
        #     input_ids=all_input_ids,
        #     attention_mask=all_attention_mask,
        # )
        # all_better_input_ids, all_worse_input_ids = all_input_ids.chunk(chunks=2, dim=0)
        # all_better_attention_mask, all_worse_attention_mask = all_attention_mask.chunk(chunks=2, dim=0)
        # all_better_sequence_log_probs, all_worse_sequence_log_probs = all_sequence_log_probs.chunk(chunks=2, dim=0)
        
        # label_smoothing_values = []
        # for sample_id in range(len(all_better_input_ids)):
        #     better_input_ids = all_better_input_ids[sample_id]
        #     better_attention_mask = all_better_attention_mask[sample_id]
            
        #     worse_input_ids = all_worse_input_ids[sample_id]
        #     worse_attention_mask = all_worse_attention_mask[sample_id]
            
        #     init_values = [all_init_value_list[sample_id * 2], all_init_value_list[sample_id * 2 + 1]]
        #     better_sequence_log_probs, worse_sequence_log_probs = all_better_sequence_log_probs[sample_id], all_worse_sequence_log_probs[sample_id]
            
        #     with torch.no_grad():
        #         torch.cuda.empty_cache()
        #         ref_better_sequence_log_probs = self.compute_log_probs(
        #             self.actor_reference_model.module,
        #             input_ids=better_input_ids.unsqueeze(0),
        #             attention_mask=better_attention_mask.unsqueeze(0),
        #         )[0]
        #         torch.cuda.empty_cache()
        #         ref_worse_sequence_log_probs = self.compute_log_probs(
        #             self.actor_reference_model.module,
        #             input_ids=worse_input_ids.unsqueeze(0),
        #             attention_mask=worse_attention_mask.unsqueeze(0),
        #         )[0]
            
        #     better_end_index = better_attention_mask.nonzero()[-1]
        #     worse_end_index = worse_attention_mask.nonzero()[-1]
        #     try:
        #         diverge_index = (better_input_ids != worse_input_ids).nonzero()[0]
        #         assert 0 <= diverge_index <= better_end_index, 'diverge index is out of range!'
        #         assert 0 <= diverge_index <= worse_end_index, 'diverge index is out of range!'
        #     except:
        #         continue
            
        #     better_seq_slice = slice(diverge_index - 1, better_end_index)
        #     worse_seq_slice = slice(diverge_index - 1, worse_end_index)
            
        #     better_log_probs = better_sequence_log_probs[better_seq_slice].sum(dim=-1)
        #     worse_log_probs = worse_sequence_log_probs[worse_seq_slice].sum(dim=-1)
        #     ref_better_log_probs = ref_better_sequence_log_probs[better_seq_slice].sum(dim=-1)
        #     ref_worse_log_probs = ref_worse_sequence_log_probs[worse_seq_slice].sum(dim=-1)
        #     better_log_ratio = better_log_probs - ref_better_log_probs
        #     worse_log_ratio = worse_log_probs - ref_worse_log_probs
        #     if self.args.norm_prob or self.args.ipo:
        #         better_log_ratio /= better_attention_mask[better_seq_slice].sum(dim=-1) ** self.args.length_penalty
        #         worse_log_ratio /= worse_attention_mask[worse_seq_slice].sum(dim=-1) ** self.args.length_penalty
        #     logits = better_log_ratio - worse_log_ratio
            
        #     if self.args.ipo:
        #         losses.append((logits - 1 / (2 * self.scale_coeff)) ** 2)
        #     elif self.args.conservative:
        #         qb, qw = init_values
        #         confidence = calculate_preference_confidence(qb, qw)
        #         label_smoothing = min(1 - confidence, 0.5)
        #         losses.append(
        #             - F.logsigmoid(self.scale_coeff * logits) * (1 - label_smoothing)
        #             - F.logsigmoid(-self.scale_coeff * logits) * label_smoothing
        #         )
        #         label_smoothing_values.append(label_smoothing)
        #     else:
        #         losses.append(-F.logsigmoid(self.scale_coeff * logits))
        #     better_sample_rewards.append(self.scale_coeff * better_log_ratio.detach())
        #     worse_sample_rewards.append(self.scale_coeff * worse_log_ratio.detach())
            
        #     max_lengths.append(better_attention_mask[start:].float().sum())
        #     max_lengths.append(worse_attention_mask[start:].float().sum())
        
        # if not len(losses): return {}
        
        # loss = torch.stack(losses).mean()
        # max_generated_length = torch.stack(max_lengths).max()
        # total_max_generated_length = max_generated_length + start
        # better_sample_rewards = torch.stack(better_sample_rewards)  # size = (B,)
        # worse_sample_rewards = torch.stack(worse_sample_rewards)  # size = (B,)
        # rewards_accuracy = (
        #     (better_sample_rewards > worse_sample_rewards).float().mean()
        # )  # size = ()
        # better_sample_rewards = better_sample_rewards.mean()  # size = ()
        # worse_sample_rewards = worse_sample_rewards.mean()  # size = ()
        # rewards = better_sample_rewards + worse_sample_rewards  # size = ()
        # rewards_margin = better_sample_rewards - worse_sample_rewards  # size = ()
        
        # torch.cuda.empty_cache()
        # self.actor_model.backward(loss)
        # self.actor_model.step()
        
        # loss = get_all_reduce_mean(loss)
        # rewards = get_all_reduce_mean(rewards)
        # better_sample_rewards = get_all_reduce_mean(better_sample_rewards)
        # worse_sample_rewards = get_all_reduce_mean(worse_sample_rewards)
        # rewards_accuracy = get_all_reduce_mean(rewards_accuracy)
        # rewards_margin = get_all_reduce_mean(rewards_margin)
        # max_generated_length = get_all_reduce_max(max_generated_length)
        # total_max_generated_length = get_all_reduce_max(total_max_generated_length)
        
        # return {
        #     'train/loss': loss.item(),
        #     'train/rewards': rewards.item(),
        #     'train/better_sample_rewards': better_sample_rewards.item(),
        #     'train/worse_sample_rewards': worse_sample_rewards.item(),
        #     'train/rewards_accuracy': rewards_accuracy.item(),
        #     'train/rewards_margin': rewards_margin.item(),
        #     'train/lr': self.actor_model.optimizer.param_groups[0]['lr'],
        #     'train/r_scores': float(prediction[0]),
        #     'train/correct': float(prediction[1]),
        #     'train/n_sample': n_sample,
        #     'train/max_generated_length': max_generated_length.item(),
        #     'train/total_max_generated_length': total_max_generated_length.item(),
        #     'train/label_smoothing': sum(label_smoothing_values) / len(label_smoothing_values) if len(label_smoothing_values) else 0,
        #     'train/cur_max_new_tokens': cur_max_new_tokens,
        # }
    
