import time

import requests
from openai import OpenAI

from treeQA_Config import nv_embed_v2_url,RetrieveModelName


def getOpenAIEmbeddings(textList):
    client = OpenAI()
    response = client.embeddings.create(
        input=textList,
        model="text-embedding-3-small"
    )
    embeddings=[]
    for item in response.data:
        embeddings.append(item.embedding)
    return embeddings

def getNVEmbeddings(textList, max_retries=3, retry_delay=3):
    url = nv_embed_v2_url
    data = {
        "text_list": textList,
        "instruction":"retrieve passages that answer the question",
        "max_length": 32768
    }
    for attempt in range(max_retries + 1):  # 尝试次数为 max_retries + 1
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()['embeddings']
            else:
                if attempt < max_retries:
                    print(f"请求失败，状态码: {response.status_code}，正在重试 ({attempt + 1}/{max_retries})...")
                    time.sleep(retry_delay)  # 等待指定的时间后再试
                else:
                    print(f"所有重试均失败，状态码: {response.status_code}")
                    print(response.text)
                    return None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                print(f"网络请求错误: {e}，正在重试 ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
            else:
                print(f"所有重试均失败，最后一次错误为: {e}")
                return None

model_functions = {
    "nv-embed-v2": getNVEmbeddings,
    "text-embedding-3-small": getOpenAIEmbeddings,
}

def getEmbeddings(textList,model_name=RetrieveModelName):
    response_function = model_functions.get(model_name)
    if response_function:
        return response_function(textList)
    else:
        raise ValueError(f"Invalid model name: {model_name}")