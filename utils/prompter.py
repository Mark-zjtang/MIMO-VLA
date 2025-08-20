"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        input: str,
        question: Union[None, str] = None,
        answer: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                input=input, question=question,
            )
        else:
            res = self.template["prompt_no_input"].format(
                input=input
            )
        if answer:

            res = f"{res}{answer}"
            # print('res_lable',res)
        if self._verbose:
            print(res)
        return res

    def get_response(self, answer: str) -> str:
        return answer.split(self.template["response_split"])[1].strip()


    def get_input(self, question: str) -> str:
        return question.split(self.template["prompt_input"])[1].strip()

    def get_output(self, question: str) -> str:
        return question.split(self.template["prompt_input"])[1].strip()

