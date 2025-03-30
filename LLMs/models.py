from openai import OpenAI
from treeQA_Config import model_name, aliApiKey, deepseekApiKey


def get_DeepSeek_Response(prompt,query):

    client = OpenAI(api_key=deepseekApiKey, base_url="https://api.deepseek.com")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            stream=False,
            max_tokens=2048,
            temperature=0.0
        )

        # 计算当前查询的 token 数量
        currentTokenCount = response.usage.total_tokens
        return response.choices[0].message.content,currentTokenCount
    except Exception as e:
        print(e)
        return "There is no answer for this question."

def get_deepseekV3(prompt, query):
    api_key = aliApiKey
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.chat.completions.create(
        model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ],
        max_tokens=4096
    )
    # 计算当前查询的 token 数量
    currentTokenCount = completion.usage.total_tokens

    # 打印当前消耗和总消耗
    
    return completion.choices[0].message.content,currentTokenCount
def get_qwen14b_ali(prompt, query):
    api_key = aliApiKey
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=api_key,  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    completion = client.chat.completions.create(
        model="qwen2.5-14b-instruct",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
        messages=[
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': query}
        ],
        max_tokens=4096
    )
    # 计算当前查询的 token 数量
    currentTokenCount = completion.usage.total_tokens

    # 打印当前消耗和总消耗
    
    return completion.choices[0].message.content,currentTokenCount


def get_gpt_response(prompt,query):
    client = OpenAI(
          base_url="https://api.gptsapi.net/v1",
         api_key="sk-yUA2cc6412572324b528da0a02dc4799f505305ce24PWEjr"
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=0.01
    )
    # 计算当前查询的 token 数量
    currentTokenCount = completion.usage.total_tokens

    # 打印当前消耗和总消耗
    
    return completion.choices[0].message.content,currentTokenCount

model_functions = {
    "deepseekV3": get_DeepSeek_Response,
    "qwen2.5-instruct-14b": get_qwen14b_ali,
    "deepseekV3_ali": get_deepseekV3,
    "gpt3.5-turbo":get_gpt_response
}
def getModelResponse(prompt, query, model_name=model_name):
    response_function = model_functions.get(model_name)
    if response_function:
        return response_function(prompt, query)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

if __name__ == "__main__":
    print(getModelResponse("You are a helpful assistance.", "Who are you?"))
