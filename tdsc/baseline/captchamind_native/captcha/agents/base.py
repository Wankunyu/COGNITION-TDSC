# Copyright Sierra

import abc
from typing import Optional
from captcha.env.base import Env
from captcha.data_types import SolveResult

class Agent(abc.ABC):
    @abc.abstractmethod
    def solve(
        self, env: Env, task_index:int, max_num_steps: int = 30
    ) -> SolveResult:
        raise NotImplementedError
