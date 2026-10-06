# encoding: utf-8

from captcha.env.base import Env
from captcha.env.rotation_match.data import load_data, get_image_path, get_tmp_file_path, set_current_sample_path
from captcha.utils import *
from captcha.utils.image_utils import *
from captcha.data_types import EnvResetResponse, EnvResponse, Action

from typing import List, Dict, Any, Optional, Union, Callable


class RotationMatchEnv(Env):
    def __init__(self, task_index: int) -> None:
        super().__init__(
            data_load_func=load_data,
            task_index=task_index
        )

        # image_path = [get_image_path(v) for v in self.task_data["image_list"]]
        # 设置当前样本路径
        current_sample_data = self.task_data
        if '_sample_path' in current_sample_data:
            set_current_sample_path(current_sample_data['_sample_path'])
        image_path = [get_image_path(v) for v in self.task_data["image_list"]]

        self.current_state = {
            "image_list": image_path,
            "ground_truth_image_index": self.task_data["ground_truth_image_index"],
            "arrow_box": self.task_data["arrow_box"],
            "submit_box": self.task_data["submit_box"]
        }
        self.current_state['current_image_index'] = 0

    def reset(self, task_index: int) -> EnvResetResponse:
        self.task_index = task_index
        self.task_data = self.data[task_index]
        self.actions = []

        # image_path = [get_image_path(v) for v in self.task_data["image_list"]]
        # 设置当前样本路径
        if '_sample_path' in self.task_data:
            set_current_sample_path(self.task_data['_sample_path'])
        image_path = [get_image_path(v) for v in self.task_data["image_list"]]

        self.current_state = {
            "image_list": image_path,
            "ground_truth_image_index": self.task_data["ground_truth_image_index"],
            "arrow_box": self.task_data["arrow_box"],
            "submit_box": self.task_data["submit_box"]
        }
        self.current_state['current_image_index'] = 0
        # 打印初始传入的图片路径
        print(f"初始传入图片路径: {image_path[0]}, current_image_index: {self.current_state['current_image_index'] + 1}")
        return EnvResetResponse(observation=[image_path[0]])

    def is_click_submit(self, pos: List[int]) -> bool:
        return is_in_box(self.current_state["submit_box"], pos)

    def is_click_left_arrow(self, pos: List[int]) -> bool:
        left_arrow_box = self.current_state["arrow_box"][0]
        return is_in_box(left_arrow_box, pos)

    def is_click_right_arrow(self, pos: List[int]) -> bool:
        right_arrow_box = self.current_state["arrow_box"][1]
        return is_in_box(right_arrow_box, pos)

    def calc_reward(self) -> float:
        ground_truth_image_index = self.current_state["ground_truth_image_index"] 
        current_image_index = self.current_state["current_image_index"] + 1
        # 打印选择结果
        print(f"模型选择的index: {current_image_index}, 真实的index: {ground_truth_image_index}")

        if ground_truth_image_index != current_image_index:
            return 0.0
        else:
            return 1.0

    def get_next_image_index(self, current_image_index: int, arrow_type: str) -> int:
        image_cnt = len(self.current_state["image_list"])
        if arrow_type == 'left':
            if current_image_index == 0:
                next_image_index = image_cnt - 1
            else:
                next_image_index = current_image_index - 1
        elif arrow_type == 'right':
            next_image_index = (current_image_index + 1) % image_cnt
        else:
            raise ValueError("arrow_type must be left or right")

        return next_image_index

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        reward = 0.0
        done = 0
        observation = []

        if action.name != 'click':
            done = 1
            return EnvResponse(observation=observation, reward=reward, done=done)
        else:
            position = action.kwargs["position"]
            current_image_index = self.current_state["current_image_index"]

            if self.is_click_left_arrow(position):
                self.current_state['current_image_index'] = self.get_next_image_index(current_image_index, 'left')
            elif self.is_click_right_arrow(position):
                self.current_state['current_image_index'] = self.get_next_image_index(current_image_index, 'right')
            elif self.is_click_submit(position):
                print("submit result!!!")
                done = 1
                reward = self.calc_reward()
            else:
                pass

            image_path = self.current_state["image_list"][self.current_state["current_image_index"]]
            observation = [image_path]
            # 打印当前传入的图片路径
            print(f"传入图片路径: {image_path}, current_image_index: {self.current_state['current_image_index'] + 1}")

            return EnvResponse(observation=observation, reward=reward, done=done)


if __name__ == "__main__":
    env = RotationMatchEnv(task_index=0)
    res = env.reset(task_index=0)