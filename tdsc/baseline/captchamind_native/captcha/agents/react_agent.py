# encoding: utf-8

import json
import os
import shutil
from datetime import datetime
import re

from captcha.agents.base import Agent
from captcha.env.base import Env
from captcha.data_types import Action, SolveResult
from captcha.utils.model_utils import *
from captcha.utils.image_utils import *
from captcha.utils import * 

CLICK_ACTION_EXAMPLE = """
{
    "name": "click",
    "arguments": {
        "position": [100, 150]
    }
}
"""

DRAG_ACTION_EXAMPLE = """
{
    "name": "drag",
    "arguments": {
        "from": [100, 150],
        "to": [100, 200]
    }
}
"""

ENTER_NUMBER_ACTION_EXAMPLE = """
{
    "name": "enter_number",
    "arguments": {
        "number": 47
    }
}
"""

#
# PROMPT_TEMPLATE = """
# You are a captcha solver and your task is to solve a captcha step by step.
# {instruction}
# At each step, your generation should have exactly the following format:
# <think>Your reasoning process to understand the task, analyze the image, and decide what action to take.</think>
# <tool_call>{{"name": <function-name>, "arguments": <args-json-object>}}</tool_call>
#
# Your available actions are click, drag and enter_number(if there is an input box in the image), examples are:
# 1. {click_action}
# 2. {drag_action}
# 3. {{"name": "bounding", "arguments": {{"boxes": [[50, 50, 150, 150], [200, 200, 300, 300]]}}}}
# 4. {enter_number_action}
#
# You should only take an action at a step. After each step, you obtain an observation. Before solving the task, you should first use bounding boxes to mark the key positions.
# You can click the submit button to submit the final result.
#  """


PROMPT_TEMPLATE = """
You are a captcha solver and your task is to solve a captcha step by step. 
{instruction}
At each step, your generation should have exactly the following format:
<think>Your reasoning process to understand the task, analyze the image, and decide what action to take.</think>
<tool_call>{{"name": <function-name>, "arguments": <args-json-object>}}</tool_call>

Your available actions are click, drag and enter_number(if there is an input box in the image), examples are:
1. {click_action}
2. {drag_action}
3. {enter_number_action}

You should only take an action at a step. After each step, you obtain an observation.

You can click the submit button to submit the final result.
"""


from typing import Optional, List, Dict, Any, Tuple

class ReactAgent(Agent):
    def __init__(
        self,
        model: str,
        instruction: str,
        temperature: float=0.0,
        debug: bool=True , # 新增
        save_images: bool=True  # 新增
    ) -> None:
        
        self.prompt = PROMPT_TEMPLATE.format(
            instruction=instruction,
            click_action=CLICK_ACTION_EXAMPLE,
            drag_action=DRAG_ACTION_EXAMPLE,
            enter_number_action=ENTER_NUMBER_ACTION_EXAMPLE
        )
        self.model = model
        self.temperature = temperature
        #  添加总计数器
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        


    def generate_next_step(
        self, 
        messages: List[Dict[str, Any]],
        step_num: int = 0
    ) -> Tuple[Dict[str, Any], Action, float]:

        res, usage = completion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        print("Raw response (res):", res)

        # 首先尝试提取 <tool_call> 标签内的内容
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        match = re.search(tool_call_pattern, res, re.DOTALL)

        if match:
            action_str = match.group(1).strip()
        else:
            # 尝试提取 <action> 标签（JSON 格式）
            action_pattern = r'<action>(.*?)</action>'
            action_match = re.search(action_pattern, res, re.DOTALL)

            if action_match:
                action_content = action_match.group(1).strip()
                # 检查是否是 JSON 格式（以 { 开头）
                if action_content.startswith('{'):
                    action_str = action_content
                else:
                    # 如果是 XML 格式，提取 <name> 和 <arguments>
                    name_pattern = r'<name>(.*?)</name>'
                    args_pattern = r'<arguments>(.*?)</arguments>'

                    name_match = re.search(name_pattern, action_content, re.DOTALL)
                    args_match = re.search(args_pattern, action_content, re.DOTALL)

                    if name_match and args_match:
                        func_name = name_match.group(1).strip()
                        args_content = args_match.group(1).strip()

                        # 提取参数（如 <position>[918, 711]</position>）
                        position_pattern = r'<position>(.*?)</position>'
                        position_match = re.search(position_pattern, args_content, re.DOTALL)

                        if position_match:
                            position_str = position_match.group(1).strip()
                            action_str = f'{{"name": "{func_name}", "arguments": {{"position": {position_str}}}}}'
                        else:
                            action_str = action_content
                    else:
                        action_str = action_content
            else:
                # 尝试提取 <function-name> 和 <args-json-object>
                func_pattern = r'<function-name>(.*?)</function-name>'
                args_pattern = r'<args-json-object>(.*?)</args-json-object>'

                func_match = re.search(func_pattern, res, re.DOTALL)
                args_match = re.search(args_pattern, res, re.DOTALL)

                if func_match and args_match:
                    func_name = func_match.group(1).strip()
                    args_str = args_match.group(1).strip()
                    # 组合成标准 JSON 格式
                    action_str = f'{{"name": "{func_name}", "arguments": {args_str}}}'
                else:
                    # 如果没找到，尝试提取 </think> 之后的内容
                    think_pattern = r'</think>\s*(.*)'
                    think_match = re.search(think_pattern, res, re.DOTALL)
                    if think_match:
                        action_str = think_match.group(1).strip()
                    else:
                        action_str = res.strip()

        # 清理格式：移除markdown代码块标记
        action_str = re.sub(r'```(?:json)?\s*', '', action_str)  # 移除 ```json 和 ```

        # 移除可能存在的 XML 结束标签
        action_str = re.sub(r'</tool_call>|</action>', '', action_str)

        # 移除 JSON 中的注释（在压缩空白之前）
        action_str = re.sub(r'#[^\n]*', '', action_str)  # 移除 # 注释
        action_str = re.sub(r'//[^\n]*', '', action_str)  # 移除 // 注释

        # 如果有多个 JSON 对象，只提取第一个（在压缩空白之前）
        if '{' in action_str:
            brace_count = 0
            first_json_end = -1
            first_json_start = action_str.index('{')

            for i in range(first_json_start, len(action_str)):
                char = action_str[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        first_json_end = i + 1
                        break

            if first_json_end > 0:
                # 检查后面是否还有内容（跳过空白字符）
                remaining = action_str[first_json_end:].strip()
                if remaining and remaining.startswith('{'):
                    print(f"⚠️ 检测到多个 JSON 对象，只提取第一个")

                action_str = action_str[first_json_start:first_json_end]

        # 最后移除多余空白
        action_str = re.sub(r'\s+', ' ', action_str)
        action_str = action_str.strip()

        try:
            action_parsed = json.loads(action_str)
            print(" JSON 解析成功:", action_parsed)

        except json.JSONDecodeError as e:
            # JSON 解析失败，尝试通过正则提取关键信息来构造 JSON
            print(f" JSON 解析失败: {e}，尝试提取关键信息...")

            # 提取 name
            name_match = re.search(r'"name"\s*:\s*"(\w+)"', action_str)
            action_name = name_match.group(1) if name_match else None

            if action_name == "click":
                # 提取 position: [x, y]
                position_match = re.search(r'"position"\s*:\s*\[?\s*(\d+)\s*,\s*(\d+)', action_str)
                if position_match:
                    x, y = int(position_match.group(1)), int(position_match.group(2))
                    action_parsed = {
                        "name": "click",
                        "arguments": {"position": [x, y]}
                    }
                    print(f" 成功提取 click 动作: position=[{x}, {y}]")
                else:
                    raise ValueError(f"无法从字符串中提取 click 动作的 position: {action_str}")

            elif action_name == "drag":
                # 提取 from: [x1, y1] 和 to: [x2, y2]
                from_match = re.search(r'"from"\s*:\s*\[?\s*(\d+)\s*,\s*(\d+)', action_str)
                to_match = re.search(r'"to"\s*:\s*\[?\s*(\d+)\s*,\s*(\d+)', action_str)
                if from_match and to_match:
                    x1, y1 = int(from_match.group(1)), int(from_match.group(2))
                    x2, y2 = int(to_match.group(1)), int(to_match.group(2))
                    action_parsed = {
                        "name": "drag",
                        "arguments": {"from": [x1, y1], "to": [x2, y2]}
                    }
                    print(f" 成功提取 drag 动作: from=[{x1}, {y1}], to=[{x2}, {y2}]")
                else:
                    raise ValueError(f"无法从字符串中提取 drag 动作的 from/to: {action_str}")

            elif action_name == "enter_number":
                # 提取 number
                number_match = re.search(r'"number"\s*:\s*(\d+)', action_str)
                if number_match:
                    number = int(number_match.group(1))
                    action_parsed = {
                        "name": "enter_number",
                        "arguments": {"number": number}
                    }
                    print(f" 成功提取 enter_number 动作: number={number}")
                else:
                    raise ValueError(f"无法从字符串中提取 enter_number 动作的 number: {action_str}")

            else:
                raise ValueError(f"无法识别动作类型: {action_name}，原始字符串: {action_str}")

        assert "name" in action_parsed
        assert "arguments" in action_parsed
        action = Action(name=action_parsed["name"], kwargs=action_parsed["arguments"])
        return res, action, usage

    def solve(
        self,
        env: Env,
        task_index: int,
        max_num_steps: int = 30
    ) -> SolveResult:

        response = env.reset(task_index=task_index)
        reward = 0.0
        #  每个样本的计数器
        sample_input_tokens = 0
        sample_output_tokens = 0
        sample_total_tokens = 0

        assert len(response.observation) > 0, "initial observation should not be empty!"

        image_message = gen_image_message(response.observation)
        messages: List[Dict[str, Any]] = [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": self.prompt
                    }
                ]
            }
        ]
        messages[0]["content"] += image_message

        info = {}
        for step in range(max_num_steps):

            message, action, usage = self.generate_next_step(messages, step + 1)

            #  累加样本级别的 token
            sample_input_tokens += usage.get('prompt_tokens', 0)
            sample_output_tokens += usage.get('completion_tokens', 0)
            sample_total_tokens += usage.get('total_tokens', 0)
            
            #  累加总计的 token
            self.total_input_tokens += usage.get('prompt_tokens', 0)
            self.total_output_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)
            response = env.step(action)
            obs = response.observation
            reward = response.reward

            if not obs:
                obs = ""
                messages.extend(
                    [
                        {"role": "assistant", "content": message},
                        {"role": "user", "content": obs},
                    ]
                )
            else:
                image_message = gen_image_message(obs)
                assistant_message = {
                    "role": "assistant", 
                    "content": message
                }
                user_message = {
                    "role": "user",
                    "content": image_message
                }
                extend_message = [assistant_message, user_message]
                messages.extend(extend_message)
                # print(messages)
                
            if response.done:
                break
        print(f" 本样本 Token 使用: 输入={sample_input_tokens}, 输出={sample_output_tokens}, 总计={sample_total_tokens}")

        return SolveResult(
            messages=messages,
            reward=reward,
        )
    def get_token_stats(self):
        """获取总 token 统计"""
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_tokens
        }
    
    def reset_token_stats(self):
        """重置 token 统计"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
