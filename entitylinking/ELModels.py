
import re
from LLMs.models import getModelResponse
from treeQA_Config import ELTop_k, el_model
import aiohttp
import asyncio
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from aiohttp import ClientTimeout
from treeQA_Config import Azure_key, Azure_endpoint, proxies,relik_server_url
import json
import requests

# Azure 客户端认证
def authenticate_client():
    ta_credential = AzureKeyCredential(Azure_key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=Azure_endpoint,
        credential=ta_credential)
    return text_analytics_client


client = authenticate_client()


# 异步获取 Wikidata ID（支持代理）
async def get_wikidata_id_with_proxy(wikipedia_url, retries=3, delay=5):
    """
    从 Wikipedia URL 获取对应的 Wikidata ID，支持代理。

    参数:
    - wikipedia_url (str): Wikipedia 页面 URL。
    - retries (int): 最大重试次数。
    - delay (int): 每次重试之间的间隔时间（秒）。

    返回:
    - str: Wikidata 实体 ID 或 None（如果多次重试失败）。
    """
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession(
                    timeout=ClientTimeout(total=30),
                    connector=aiohttp.TCPConnector(ssl=False)  # 禁用 SSL 验证（视需求开启）
            ) as session:
                async with session.get(
                        f"https://www.wikidata.org/w/api.php",
                        params={
                            "action": "wbgetentities",
                            "sites": "enwiki",
                            "titles": wikipedia_url.split("/")[-1],
                            "format": "json",
                        },
                        proxy=proxies['http']  # 使用代理
                ) as response:
                    data = await response.json()
                    entities = data.get("entities", {})
                    return next(iter(entities.keys()), None)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {wikipedia_url}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
    return None


# 主函数：提取实体并获取 Wikidata ID
async def azureEntityLinking(text):
    """
    Performs Azure Entity Linking and fetches Wikidata IDs, returning results as a JSON string.
    """
    try:
        documents = [text]
        # 注意：实际 SDK 可能返回一个迭代器或列表，确保你正确地获取第一个结果
        results = client.recognize_linked_entities(documents=documents)
        result = None
        # 处理结果迭代器（如果适用）
        for doc_result in results:
            if not doc_result.is_error:
                result = doc_result
                break # 只处理第一个文档的结果
            else:
                # 处理文档级别的错误
                print(f"Error processing document: {doc_result.id}, Error: {doc_result.error}")
                # 可以选择抛出异常或返回错误JSON
                error_info = {"error": f"Document level error: {doc_result.error.code}", "message": doc_result.error.message}
                return json.dumps(error_info, ensure_ascii=False)

        if result is None:
             # 如果没有成功处理任何文档
            return json.dumps({"error": "No document processed successfully", "entities": []}, ensure_ascii=False)


        entities_data = [] # 存储最终的实体信息字典
        tasks = []         # 存储异步获取 Wikidata ID 的任务

        for entity in result.entities:
            # 创建实体信息字典
            entity_info = {
                "text": entity.name,
                "Wikipedia URL": entity.url,
                "definition": entity.name,
                "wikidata": None  # 先设置为 None
            }
            entities_data.append(entity_info)
            # 创建获取 Wikidata ID 的任务
            tasks.append(get_wikidata_id_with_proxy(entity.url))

        # 并发执行所有获取 Wikidata ID 的任务
        # asyncio.gather 会保持结果的顺序与输入任务的顺序一致
        wikidata_ids = await asyncio.gather(*tasks, return_exceptions=True) # return_exceptions=True 可以在任务失败时不中断其他任务

        # 将获取到的 Wikidata ID (或异常) 填入对应的实体字典中
        for i, wikidata_id_or_exception in enumerate(wikidata_ids):
            if isinstance(wikidata_id_or_exception, Exception):
                # 处理获取 Wikidata ID 时的错误
                print(f"Error fetching Wikidata ID for {entities_data[i]['Wikipedia URL']}: {wikidata_id_or_exception}")
                entities_data[i]["wikidata"] = f"Error: {wikidata_id_or_exception}" # 或者设置为 None 或其他错误标记
            else:
                entities_data[i]["wikidata"] = wikidata_id_or_exception

        # 2. 使用 json.dumps 将 Python 列表转换为 JSON 字符串
        #    indent=4: 使输出的 JSON 字符串具有缩进，更易读（可选）
        #    ensure_ascii=False: 允许非 ASCII 字符（如中文）直接输出，而不是转义成 \uXXXX（推荐）
        return json.dumps(entities_data, indent=4, ensure_ascii=False)

    except Exception as e:
        # 处理在函数执行过程中可能发生的其他异常
        print(f"An error occurred in azureEntityLinking: {e}")
        # 返回一个表示错误的 JSON 字符串
        error_info = {"error": str(e), "entities": []}
        return json.dumps(error_info, ensure_ascii=False)

    except Exception as err:
        raise RuntimeError(f"Error extracting entities: {err}")



def relikEntityLinking(query):
    # URL 和查询参数
    url = relik_server_url
    params = {
        'text': query,
        'is_split_into_words': 'false',
        'retriever_batch_size': '32',
        'reader_batch_size': '32',
        'return_windows': 'false',
        'use_doc_topic': 'false',
        'annotation_type': 'char',
        'relation_threshold': '0.5'
    }

    # 设置请求头
    headers = {
        'accept': 'application/json'
    }

    # 发送 GET 请求
    response = requests.get(url, params=params, headers=headers)

    # 输出响应内容
    #print(response.json())
    # 遍历 JSON 数据中的每个条目
    # 遍历响应中的每个项目
    candidates_info=[]
    for item in response.json():
        # 检查'candidates'键是否存在并且是字典类型
        if 'candidates' in item and isinstance(item['candidates'], dict) and 'span' in item['candidates']:
            # 遍历每个候选者
            for candidate in item['candidates']['span'][0][0]:
                # 提取需要的信息
                text = candidate.get('text', 'N/A')
                wikidata = candidate.get('metadata', {}).get('wikidata', 'N/A')
                definition = candidate.get('metadata', {}).get('definition', 'N/A')

                # 将提取的信息添加到列表中
                candidates_info.append({
                    'text': text,
                    'wikidata': wikidata,
                    'definition': definition
                })

    # 返回JSON格式的数据
    return json.dumps(candidates_info, ensure_ascii=False, indent=4)






#通过llm进行实体抽取
def llmForEntityExtract(query,logicTree):
    prompt = """ I need you to help me understand the user's request and identify the entities involved in the user's query. Your task is simple: recognize the user's request and generate the parameters accordingly. Please note that when extracting, only the noun form is required! No plural forms or other forms of nouns, etc.
    Here's an example:
    Input: Who composed the music for Manru? The music for Manru was composed by Ignacy Jan Paderewski.
    Output: "Manru", "Ignacy Jan Paderewski"
    Input:Find information about Kirill Eskov's biography or personal details to determine his country of citizenship.
    Output:"Kirill Eskov"
    Now the user input is: """

    item_string,tokenCount = getModelResponse(prompt, query)
    logicTree.tokenCount +=tokenCount
    print("Entity extract complete!")
    # currentCount = currentCount + tokenCount
    # print(f"总token消耗：[{currentCount}]，实体抽取消耗：{tokenCount}")
    # 使用正则表达式来匹配被双引号包围的内容，包括空字符串
    pattern = r'"(.*?)"'
    # 查找所有匹配实体并将结果放入列表
    matches = re.findall(pattern, item_string)
    return matches


# 使用大语言模型进行挑选有用实体，输出实体ID
def llmForEntityFilter(itemInfo,query,logicTree):
    itemSelectPrompt = f"""Now I need you to select the entity that is truly relevant to the query.  
    I need you to select the most relevant entity ID based on the following information, 
    Output the related IDs no more than {ELTop_k} without outputting any other content.
    format: ['','','']
    entity info: {itemInfo}
    The user's query is:{query}
    """
    top1_item_string,tokenCount = getModelResponse(itemSelectPrompt, "Please begin to choose.")
    logicTree.tokenCount+=tokenCount
    # currentCount = currentCount + tokenCount
    # print(f"总token消耗：[{currentCount}]，实体过滤消耗：{tokenCount}")
    # 正则表达式模式：匹配单引号中的文本
    pattern = r"'(.*?)'"
    # 查找所有匹配项
    matches = re.findall(pattern, top1_item_string)
    print(f"Selected {len(matches)} entities!")
    return matches




def linkEntity(query, model_name=el_model):
    if model_name =="relik":
        return relikEntityLinking(query)
    return asyncio.run(azureEntityLinking(query))


# 示例调用
if __name__ == "__main__":
    input_text = """To Catch a Predator was devoted to impersonating people below the age of consent for which in North America varies by what?"""
    print(linkEntity(input_text))
    print(relikEntityLinking(input_text))