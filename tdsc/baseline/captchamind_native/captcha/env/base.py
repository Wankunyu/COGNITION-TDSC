# encoding: utf-8

from typing import List, Dict, Any, Optional, Union, Callable
from captcha.data_types import Action, Task, EnvResponse, EnvResetResponse

class Env(object):
    def __init__(
        self,
        data_load_func: Callable[[], List[Any]],
        task_index: int
    ) -> None:
        super().__init__()
        self.data_load_func = data_load_func
        self.data = data_load_func()

        self.task_index = task_index
        self.task_data = self.data[task_index]
        
        self.actions: List[Action] = []

    def reset(self, task_index: int) -> EnvResetResponse:
        # self.task_index = task_index
        # self.task = self.tasks[task_index]
        # self.actions = []
        raise NotImplementedError("reset function should be implemented")


    def step(self, action: Action) -> EnvResponse:
        # 1. take step
        # 2. update state
        # 3. check if finished
        # 4. return observation and reward
        raise NotImplementedError("reset function should be implemented")

