"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar
from datasets import load_dataset, concatenate_datasets, Dataset

from mcts_rl.datasets.base import RawDataset, RawSample
from mcts_rl.utils import extract_answer, get_math_data, list_to_dict, get_arithmo_data
import json
from tqdm import tqdm


__all__ = [
    'MATHDataset',
    'MATHTrainDataset',
    'MATHTestDataset',
    'MATHSFTTrainDataset',
    'MATHSFTTestDataset',
]
MATH_PROMPT = ''
PRM_DATA_DIR = '/workspace/MCTS-DPO/mcts_rl/datasets/raw/prm800k/prm800k/data'
MATH_DATA_DIR = '/workspace/MCTS-DPO/mcts_rl/datasets/raw/data/math'

class MATHDataset(RawDataset):
    SPLIT: ClassVar[str]
    DTYPE: ClassVar[str]

    def __init__(self) -> None:
        if MATH_DATA_DIR is not None:
            dataset = Dataset.load_from_disk(os.path.join(MATH_DATA_DIR, self.SPLIT))
            if self.SPLIT == 'test':
                dataset = dataset.shuffle(seed=42)
                dataset = dataset.select(range(100))
            if self.SPLIT == 'train':
                dataset = dataset.shuffle(seed=42)
                dataset = dataset.select(range(7500))
            self.data = dataset.sort("level")
            unique_levels = sorted(set(self.data["level"]))
            shuffled_data_per_level = []
            for lvl in unique_levels:
                subset_lvl = self.data.filter(lambda x: x["level"] == lvl)
                print(f'Level {lvl}: {len(subset_lvl)}')
                subset_lvl_shuffled = subset_lvl.shuffle(seed=42)  # optional seed for reproducibility
                shuffled_data_per_level.append(subset_lvl_shuffled)
            # 3) Concatenate them in ascending order of level:
            self.data = concatenate_datasets(shuffled_data_per_level)
            new_data = []
            for ds in tqdm(self.data):
                ds['step_solution'] = []
                cnt = 0
                for step in ds['gold_solution_steps']:
                    cnt += 1
                    ds['step_solution'].append(f'## Step {cnt}: ' + step + '\n\n')
                ds['step_solution'].append('The final answer is: $\\boxed{' + ds['answer'] +'}$')
                new_data.append(ds)
            self.data = new_data

        else:
            # self.data = load_dataset('hendrycks/competition_math', split=self.SPLIT, trust_remote_code=True)
            subsets = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
            datasets = []
            for subset in subsets:
                ds = load_dataset("EleutherAI/hendrycks_math", subset, split=self.SPLIT, trust_remote_code=True)
                datasets.append(ds)
            combined_dataset = concatenate_datasets(datasets)
            if self.SPLIT == 'test':
                combined_dataset = combined_dataset.shuffle(seed=42)
                combined_dataset = combined_dataset.select(range(100))
            # sort by level
            self.data = combined_dataset.sort("level")
            # 1) Get unique levels:
            unique_levels = sorted(set(self.data["level"]))
            # 2) For each level, filter the dataset and shuffle the subset:
            shuffled_data_per_level = []
            for lvl in unique_levels:
                subset_lvl = self.data.filter(lambda x: x["level"] == lvl)
                subset_lvl_shuffled = subset_lvl.shuffle(seed=42)  # optional seed for reproducibility
                shuffled_data_per_level.append(subset_lvl_shuffled)

            # 3) Concatenate them in ascending order of level:
            self.data = concatenate_datasets(shuffled_data_per_level)

            # combine MATH and PRM data
            with open(os.path.join(PRM_DATA_DIR, 'phase1_train.jsonl'), 'r') as f:
                json_file = list(f)
            result_list = []
            for json_str in json_file:
                result = json.loads(json_str)
                result_list.append(result)
            with open(os.path.join(PRM_DATA_DIR, 'phase2_train.jsonl'), 'r') as f:
                json_file = list(f)
            for json_str in json_file:
                result = json.loads(json_str)
                result_list.append(result)
            new_data = []
            # additional_test = False
            for ds in tqdm(self.data):
                for idx, result in  enumerate(result_list):
                    if ds['problem'] == result['question']['problem'] and result['label']['finish_reason'] == 'solution':

                        # if result['question'].get('pre_generated_answer', None) is not None:
                        #     pre_gen = result['question'].get('pre_generated_answer', None)
                        #     gen_ans = result['question'].get('ground_truth_answer', None)
                        #     if extract_answer(pre_gen) != extract_answer(gen_ans):
                        #         print('additional test failed')
                        #         continue
                        ds['step_solution'] = []
                        cnt = 0
                        for step in result['label']['steps']:
                            for candidate in step['completions']:
                                if candidate['rating'] == 1:
                                    cnt += 1
                                    ds['step_solution'].append(f'## Step {cnt} ' + candidate['text'] + '\n\n')
                                    break
                        ds['answer'] = result['question']['ground_truth_answer']  
                        ds['step_solution'].append('The final answer is: $\\boxed{' + result['question']['ground_truth_answer'] +'}$')
                        break
                # if 'step_solution' in ds:
                new_data.append(ds)
            self.data = new_data
            if self.DTYPE == 'arithmo':
                math_dict = list_to_dict(self.data)
                arithmo_dict = list_to_dict(get_math_data(load_dataset('akjindal53244/Arithmo-Data', split=self.SPLIT)))
                arithmo = {k:v for k, v in arithmo_dict.items() if k in math_dict}
                self.data = [vv for v in arithmo.values() for vv in v]
                # self.data = get_arithmo_data(arithmo)
                


    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        solution = data['solution']
        answer = extract_answer(solution)
        step_solution = data.get('step_solution', [])
        gt_answer = data.get('answer', None)
        # answer = data['answer']
        # if gt_answer != answer:
        #     step_solution = []

        if self.DTYPE == 'default':
            # solution = f'{solution}\nThe answer is {answer}'
            solution = f'{solution}\nThe answer is {gt_answer}'
        return RawSample(
            instructions=MATH_PROMPT,
            input=data['problem'] if 'problem' in data else data['question'],
            answer=solution,
            # final_answer=answer,
            final_answer=gt_answer if gt_answer is not None else answer,
            # final_answer_content=answer,
            final_answer_content=gt_answer if gt_answer is not None else answer,
            step_solution=step_solution
        )

    def __len__(self) -> int:
        return len(self.data)


class MATHTrainDataset(MATHDataset):
    NAME: str = 'MATH/train'
    SPLIT: str = 'train'
    DTYPE: str = 'default'


class MATHTestDataset(MATHDataset):
    NAME: str = 'MATH/test'
    SPLIT: str = 'test'
    DTYPE: str = 'default'


class MATHSFTTrainDataset(MATHDataset):
    NAME: str = 'MATHSFT/train'
    SPLIT: str = 'train'
    DTYPE: str = 'arithmo'


class MATHSFTTestDataset(MATHDataset):
    NAME: str = 'MATHSFT/test'
    SPLIT: str = 'test'
    DTYPE: str = 'arithmo'
