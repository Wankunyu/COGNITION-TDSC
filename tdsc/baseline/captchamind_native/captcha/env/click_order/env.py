# encoding: utf-8

from captcha.env.base import Env
from captcha.env.click_order.data import load_data, get_image_path, get_tmp_file_path, set_current_sample_path, load_number_images
from captcha.utils import *
from captcha.utils.image_utils import *
from captcha.data_types import EnvResetResponse, EnvResponse, Action

from typing import List, Dict, Any, Optional, Union, Callable


class ClickOrderEnv(Env):

    def __init__(
        self,
        task_index: int
    ) -> None:
        super().__init__(
            data_load_func=load_data,
            task_index=task_index
        )
        
        # 设置当前样本路径
        current_sample_data = self.task_data
        if '_sample_path' in current_sample_data:
            set_current_sample_path(current_sample_data['_sample_path'])
            
        image_path = get_image_path(self.task_data["image"])
        
        # 加载统一的数字图片
        number_image_path = load_number_images()

        self.current_state = {
            "image": image_path,
            "number_image": number_image_path,
            "gt_box_list": self.task_data["gt_box_list"],
            "submit_box": self.task_data["submit_box"],
            "current_image": image_path  # 追踪当前显示的图片路径
        }
        
        # 打印初始传入的图片路径
        print(f"初始传入图片路径: {image_path}")
        
    def reset(self, task_index: int) -> EnvResetResponse:
        self.task_index = task_index
        self.task_data = self.data[task_index]
        self.actions = []
        
        # 设置当前样本路径
        if '_sample_path' in self.task_data:
            set_current_sample_path(self.task_data['_sample_path'])
            
        image_path = get_image_path(self.task_data["image"])
        
        # 加载统一的数字图片
        number_image_path = load_number_images()
                    
        self.current_state = {
            "image": image_path,
            "number_image": number_image_path,
            "gt_box_list": self.task_data["gt_box_list"],
            "submit_box": self.task_data["submit_box"],
            "current_image": image_path  # 追踪当前显示的图片路径
        }
        
        # 打印当前传入图片路径
        print(f"传入图片路径: {image_path}")
        return EnvResetResponse(observation=[image_path])

    def is_click_submit(self, pos: List[int]) -> bool:
        return is_in_box(self.current_state["submit_box"], pos)

    def calc_reward(self) -> float:
        gt_box_list = self.current_state["gt_box_list"]
        
        # 排除最后一个submit动作，只检查前面的点击动作
        click_actions = self.actions[:-1]  # 去掉最后一个submit动作
        
        if len(click_actions) != len(gt_box_list):
            print(f"点击动作数量不匹配: {len(click_actions)} vs {len(gt_box_list)}")
            return 0.0

        for i, action in enumerate(click_actions):
            if action.name != 'click':
                print(f"第{i+1}个动作不是点击: {action.name}")
                return 0.0

            if not is_in_box(gt_box_list[i], action.kwargs["position"]):
                print(f"第{i+1}个点击位置错误: {action.kwargs['position']} 不在 {gt_box_list[i]} 内")
                return 0.0

        print(f"模型成功完成所有{len(gt_box_list)}个点击!")
        return 1.0

    def add_num_to_image(self, pos: List[int], number: int, step: int) -> str:
        output_image_path = get_tmp_file_path("tmp_image_{}.png".format(step))
        
        # 确保有足够的数字图片
        if len(self.current_state["number_image"]) >= number:
            number_image_path = self.current_state["number_image"][number - 1]
        else:
            # 如果没有足够的数字图片，使用第一个数字图片作为默认
            if self.current_state["number_image"]:
                number_image_path = self.current_state["number_image"][0]
            else:
                # 如果完全没有数字图片，直接复制原图
                import shutil
                shutil.copy2(self.current_state["current_image"], output_image_path)
                return output_image_path
        
        box = [pos[0] - 20, pos[1] - 20, pos[0] + 20, pos[1] + 20]
        
        # 重要：在当前图片基础上添加数字，而不是原始图片
        add_to_image(
            self.current_state["current_image"],  # 使用当前图片而不是原始图片
            [box],
            number_image_path,
            output_image_path
        )
        
        # 更新当前图片路径
        self.current_state["current_image"] = output_image_path

        return output_image_path

    def step(self, action: Action) -> EnvResponse:
        self.actions.append(action)
        reward = 0.0
        done = 0
        observation = []
        
        if action.name != 'click':
            print(f"收到非点击动作: {action.name}")
            done = 1
            return EnvResponse(observation=observation, reward=reward, done=done)
        else:
            position = action.kwargs["position"]
            print(f"点击位置: {position}")
            
            if self.is_click_submit(position):
                print("submit result!!!")
                done = 1
                reward = self.calc_reward()
            else:
                result_image_path = self.add_num_to_image(position, len(self.actions), len(self.actions))
                observation = [result_image_path]
                print(f"添加数字后的图片路径: {result_image_path}")
            
            return EnvResponse(observation=observation, reward=reward, done=done)


if __name__ == "__main__":
    env = ClickOrderEnv(task_index=0)
    res = env.reset(task_index=0)