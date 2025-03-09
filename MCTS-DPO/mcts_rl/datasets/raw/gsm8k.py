"""MATH datasets."""

from __future__ import annotations

import os
from typing import ClassVar
from datasets import load_dataset

from mcts_rl.datasets.base import RawDataset, RawSample
from mcts_rl.utils import extract_answer, list_to_dict, get_math_data


__all__ = [
    'GSM8KDataset',
    'GSM8KTrainDataset',
    'GSM8KTestDataset',
    'GSM8KPoTTrainDataset',
    'GSM8KPoTTestDataset',
    'GSM8KSFTTrainDataset',
    'GSM8KSFTTestDataset',
]
# MATH_PROMPT = """You are a helpful AI assistant that provides step-by-step solutions to math problems, explaining the reasoning and including relevant code where necessary. Below are a few-shot examples of how you should structure your solutions.\n\n-------------------------\nExample 1\nQuery: Given a regular tetrahedron \\( P-ABC \\) with edge length 1, point \\( D \\) is the midpoint of \\( PC \\). Point \\( E \\) is a moving point on segment \\( AD \\). Determine the range of values for the angle formed between line \\( BE \\) and plane \\( ABC \\).\n\nResponse:\n<code>\nimport sympy as sp\n\n# Define the edge length of the tetrahedron\nedge_length = 1\n\n# Define the coordinates of the vertices of the tetrahedron\nA = sp.Matrix([0, 0, 0])\nB = sp.Matrix([1, 0, 0])\nC = sp.Matrix([0.5, sp.sqrt(3)/2, 0])\nP = sp.Matrix([0.5, sp.sqrt(3)/6, sp.sqrt(6)/3])\n\n# Define the midpoint D of PC\nD = (P + C) / 2\n<end_of_step>\n\n# Define the parametric equation of line AD\nt = sp.symbols(\'t\')\nE = A + t * (D - A)\n\n# Define the normal vector of plane ABC\nnormal_ABC = (B - A).cross(C - A)\n\n# Define the vector BE\nBE = E - B\n\n# Define the angle between BE and plane ABC\nangle = sp.asin(BE.dot(normal_ABC) / (BE.norm() * normal_ABC.norm()))\n<end_of_step>\n\n# Determine the range of the angle\n# Since E is on AD, t ranges from 0 to 1\n# The angle will be minimum when t=0 (E=A) and maximum when t=1 (E=D)\nangle_min = sp.limit(angle, t, 0)\nangle_max = sp.limit(angle, t, 1)\n<end_of_step>\n\n# Now print the final answer\nprint(f"Range of values for the angle: [{angle_min}, {angle_max}]")\n<end_of_code>\n<output>Range of values for the angle: [0, asin(sqrt(2)/3)]<end_of_output>\n<answer>The range of values for the angle formed between line \\( BE \\) and plane \\( ABC \\) is \\boxed{[0, \\arcsin(\\frac{\\sqrt{2}}{3})]}<end_of_answer>"""
MATH_PROMPT = ''

class GSM8KDataset(RawDataset):
    SPLIT: ClassVar[str]
    PTYPE: ClassVar[str]
    DTYPE: ClassVar[str]

    def __init__(self) -> None:
        if self.PTYPE != 'pot':
            self.data = load_dataset('openai/gsm8k', 'main', split=self.SPLIT, trust_remote_code=True)
        else:
            raise ValueError('Do not Support PoT for now.')
        if self.DTYPE == 'arithmo':
            gsm_dict = list_to_dict(self.data)
            arithmo_dict = list_to_dict(get_math_data(load_dataset('akjindal53244/Arithmo-Data', split=self.SPLIT)))
            arithmo = {k:v for k, v in arithmo_dict.items() if k in gsm_dict}
            self.data = [vv for v in arithmo.values() for vv in v]
            # self.data = get_arithmo_data(arithmo)

    def __getitem__(self, index: int) -> RawSample:
        data = self.data[index]
        prompt = data['problem'] if 'problem' in data else data['question']
        prompt = prompt + '\nWrite a Python program to solve this.' if self.PTYPE == 'pot' else prompt
        solution = data['solution'] if 'solution' in data else data['answer']
        answer = extract_answer(solution)
        if self.DTYPE == 'default':
            solution = f'{solution}\nThe answer is {answer}'
        return RawSample(
            instructions=MATH_PROMPT,
            input=prompt,
            answer=solution,
            final_answer=answer,
            final_answer_content=answer,
        )

    def __len__(self) -> int:
        return len(self.data)


class GSM8KSFTTrainDataset(GSM8KDataset):
    NAME: str = 'GSM8KSFT/train'
    SPLIT: str = 'train'
    PTYPE: str = 'cot'
    DTYPE: str = 'arithmo'


class GSM8KSFTTestDataset(GSM8KDataset):
    NAME: str = 'GSM8KSFT/test'
    SPLIT: str = 'test'
    PTYPE: str = 'cot'
    DTYPE: str = 'arithmo'


class GSM8KTrainDataset(GSM8KDataset):
    NAME: str = 'GSM8K/train'
    SPLIT: str = 'train'
    PTYPE: str = 'cot'
    DTYPE: str = 'default'


class GSM8KTestDataset(GSM8KDataset):
    NAME: str = 'GSM8K/test'
    SPLIT: str = 'test'
    PTYPE: str = 'cot'
    DTYPE: str = 'default'


class GSM8KPoTTrainDataset(GSM8KDataset):
    NAME: str = 'GSM8KCode/train'
    SPLIT: str = 'train'
    PTYPE: str = 'pot'
    DTYPE: str = 'default'


class GSM8KPoTTestDataset(GSM8KDataset):
    NAME: str = 'GSM8KCode/test'
    SPLIT: str = 'test'
    PTYPE: str = 'pot'
    DTYPE: str = 'default'