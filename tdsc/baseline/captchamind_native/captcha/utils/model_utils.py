# encoding: utf-8

import requests

AIDC_URL = "http://your-real-url"  # 离线

# 客服
AIDC_TOKEN = 'your-key'

headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + AIDC_TOKEN
}

AIDC_MODEL_MAPPING = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",

    # claude
    "claude-35-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-37-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    # qwen
    "qwen-max": "qwen-max-2025-01-25",
    "qwen-7b-sft": "qwen2.5_VL_7B_sft_captcha",  
    "qwen-3b-sft": "qwen2.5_VL_3B_sft_captcha",
    "qwen-7b": "Qwen2.5-VL-7B-Instruct",  
    "qwen-3b": "Qwen2.5-VL-3B-Instruct",
}



def send_request_by_prompt(prompt_text, model):
    model = AIDC_MODEL_MAPPING[model]
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(AIDC_URL, headers=headers, json=data)

    return response.json()["choices"][0]['message']['content']


def send_request_by_message(messages, model, temperature=None):
    model = AIDC_MODEL_MAPPING[model]
    data = {
        "model": model,
        "messages": messages
    }
    if temperature is not None:
        data["temperature"] = temperature

    response = requests.post(AIDC_URL, headers=headers, json=data)
    # print(response.json())
    return response.json()["choices"][0]['message']['content']

def completion(model, messages, temperature):
    try:
        model_name = AIDC_MODEL_MAPPING[model]
        data = {
            "model": model_name,
            "messages": messages
        }
        if temperature is not None:
            data["temperature"] = temperature
        
        response = requests.post(AIDC_URL, headers=headers, json=data)
        response_json = response.json()
        
        # ✅ 提取 usage 信息
        usage = response_json.get("usage", {})
        content = response_json["choices"][0]['message']['content']
        
        # 返回内容和 token 使用情况
        return content, usage
        
    except Exception as e:
        print(e)
        return None, {}