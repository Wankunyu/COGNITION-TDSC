# encoding: utf-8

from captcha.env.base import Env
from captcha.env.dice_count.data import load_data, get_image_path, get_tmp_file_path, set_current_sample_path
from captcha.utils import *
from captcha.utils.image_utils import *
from captcha.data_types import EnvResetResponse, EnvResponse, Action

from typing import List, Dict, Any, Optional, Union, Callable


class DiceCountEnv(Env):
    def __init__(self, task_index: int) -> None:
        super().__init__(
            data_load_func=load_data,
            task_index=task_index
        )

        # image_path = get_image_path(self.task_data["image"])
        # 设置当前样本路径
        current_sample_data = self.task_data
        if '_sample_path' in current_sample_data:
            set_current_sample_path(current_sample_data['_sample_path'])

        image_path = get_image_path(self.task_data["image"])

        self.white_box_path = get_image_path('white_box.png')   # 用来遮盖输入框的白色图
        self.number_image_path = []
        for digit in range(10):
            relevant_path = 'digit/digit_{}.png'.format(digit)
            self.number_image_path.append(get_image_path(relevant_path))

        self.current_state = {
            "image": image_path,
            "ground_truth": self.task_data["ground_truth"],
            "input_box": self.task_data["input_box"],
            "submit_box": self.task_data["submit_box"]
        }
        self.current_state["input_box_avaliable"] = True    # 输入框是否可输入数值
        self.current_state["input_number"] = None

    def reset(self, task_index: int) -> EnvResetResponse:
        self.task_index = task_index
        self.task_data = self.data[task_index]
        self.actions = []

        # image_path = get_image_path(self.task_data["image"])
        # 设置当前样本路径
        if '_sample_path' in self.task_data:
            set_current_sample_path(self.task_data['_sample_path'])

        image_path = get_image_path(self.task_data["image"])

        self.number_image_path = []
        for digit in range(10):
            relevant_path = 'digit/digit_{}.png'.format(digit)
            self.number_image_path.append(get_image_path(relevant_path))

        self.current_state = {
            "image": image_path,
            "ground_truth": self.task_data["ground_truth"],
            "input_box": self.task_data["input_box"],
            "submit_box": self.task_data["submit_box"]
        }
        self.current_state["input_box_avaliable"] = True    # 输入框是否可输入数值
        self.current_state["input_number"] = None
        
        # 打印初始传入的图片路径
        print(f"初始传入图片路径: {image_path}")

        return EnvResetResponse(observation=[image_path])

    def is_click_submit(self, pos: List[int]) -> bool:
        return is_in_box(self.current_state["submit_box"], pos)

    def is_click_input_box(self, pos: List[int]) -> bool:
        return is_in_box(self.current_state["input_box"], pos)

    def get_input_number_img(self, number: int) -> str:
        output_image_path = get_tmp_file_path("number_{}.png".format(number))

        digit_ones = number % 10
        digit_tens = number // 10
        digit_ones_image_path = self.number_image_path[digit_ones]
        digit_tens_image_path = self.number_image_path[digit_tens]
        if digit_tens == 0:
            return digit_ones_image_path
        else:
            input_number_image = combine_number_images(digit_tens_image_path, digit_ones_image_path)
            input_number_image.save(output_image_path)

            return output_image_path

    def add_num_to_image(self, number: int) -> str:
        input_box = self.current_state["input_box"]

        output_image_path = get_tmp_file_path("tmp_image.png")

        # 遮盖输入框
        temp_image_with_white_box_path = get_tmp_file_path("tmp_image_with_white_box.png")
        white_mask_box = [input_box[0] + 5, input_box[1] + 5, input_box[2] - 5, input_box[3] - 5]
        add_to_image(
            self.current_state["image"],
            [white_mask_box],
            self.white_box_path,
            temp_image_with_white_box_path,
            opacity=1.0
        )

        input_number_image_path = self.get_input_number_img(number)
        height = input_box[3] - input_box[1] - 10
        width = height
        input_number_box = [input_box[0] + 10, input_box[1] + 10, input_box[0] + width, input_box[1] + height]
        add_to_image(
            temp_image_with_white_box_path,
            [input_number_box],
            input_number_image_path,
            output_image_path
        )

        return output_image_path

    def calc_reward(self) -> float:
        ground_truth = self.current_state["ground_truth"]
        input_number = self.current_state["input_number"]

        # 打印选择结果
        print(f"模型输入的数字: {input_number}, 真实答案: {ground_truth}")

        if not self.current_state["input_number"] or self.current_state["input_number"] != ground_truth:
            return 0.0
        else:
            return 1.0

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        reward = 0.0
        done = 0
        observation = []

        if action.name == 'click':
            position = action.kwargs["position"]

            if self.is_click_input_box(position):
                print('click input box')
                self.current_state["input_box_avaliable"] = True
            elif self.is_click_submit(position):
                print("submit result!!!")
                done = 1
                reward = self.calc_reward()
            else:
                print('click other place')
                self.current_state["input_box_avaliable"] = False
        elif action.name == 'enter_number':
            result_image_path = self.add_num_to_image(int(action.kwargs["number"]))
            observation = [result_image_path]
            self.current_state["input_number"] = int(action.kwargs["number"])
        else:   # 错误的操作
            done = 1

        return EnvResponse(observation=observation, reward=reward, done=done)


if __name__ == "__main__":
    env = DiceCountEnv(task_index=0)
    res = env.reset(task_index=0)
    env.add_num_to_image(number=9)

