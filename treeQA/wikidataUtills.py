import json
import os
import time

import requests

from LLMs.models import getModelResponse
from treeQA_Config import proxies

from treeQA.tree_class.infoBox import infoBox
import asyncio
from aiohttp import ClientSession

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}


def searchWikiID(query, language='en', limit=2):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': {query},  # 搜索文本
        'language': {language},  # 查询语言（英文）
        'type': 'item',
        'limit': {limit}  # 返回最大数目
    }

    # 访问
    re_json = safe_request(url, params,proxies)
    # 转为json数据

    #print(json.dumps(re_json, indent=2))
    return re_json["search"]



def getWikidataEntity(entityLabels):
    results = {}
    # 通过wikidata接口获取直接检索标签
    for item in entityLabels:
        # 查询实体相关信息
        jsonData = searchWikiID(item)
        if jsonData:
            # 解析JSON字符串为Python对象
            data = jsonData
            # 遍历数据并提取'label'和'description'
            for tempItem in data:
                # print(f"getItem:{item}")
                id = tempItem.get('id')
                label = tempItem.get('label')
                description = tempItem.get('description')
                if label and description:
                    results[id] = {'text': label, 'wikidata': id, 'definition': description}
    return results

def safe_request(url, params, my_proxies, max_retries=3):

    for attempt in range(max_retries):
        try:
            response = requests.get(url,headers=HEADERS, params=params,proxies=my_proxies)
            response.raise_for_status()  # 如果响应状态码不是200，则抛出HTTPError
            return response.json()
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectionError) as e:
            if attempt < (max_retries - 1):  # i.e. if it's not the last retry
                print(f"Request failed: {e}. Retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                raise  # 如果所有重试都失败了，就抛出最后一次异常
    return None  # 如果没有异常抛出，返回None

async def fetch_relation_value(session, entity_code, relation_code, pointing):
    MAX_RETRIES = 3
    RETRY_DELAY = 3  # 秒

    if pointing:
        sparql_query = (
            f"""
            SELECT ?value ?valueLabel WHERE {{
                ?value wdt:{relation_code} wd:{entity_code}.
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,[AUTO_LANGUAGE]" }}
            }} LIMIT 10
            """
        )
    else:
        sparql_query = (
            f"""
            SELECT ?value ?valueLabel WHERE {{
              wd:{entity_code} wdt:{relation_code} ?value .
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,[AUTO_LANGUAGE]" }}
            }} LIMIT 10
            """
        )

    url = 'https://query.wikidata.org/sparql'
    params = {'query': sparql_query, 'format': 'json'}
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with session.get(url, params=params, headers=HEADERS, proxy=proxies['http']) as response:
                if response.status == 429:
                    print(f"Rate limited, waiting for {RETRY_DELAY} seconds...")
                    await asyncio.sleep(RETRY_DELAY)
                    retries += 1
                elif response.status == 200:
                    data = await response.json()
                    if 'results' in data and 'bindings' in data['results'] and data['results']['bindings']:
                        if len(data['results']['bindings'][0])==1:
                            binding = data['results']['bindings'][0]
                            if 'valueLabel' in binding and binding['valueLabel']['value']:
                                return binding['valueLabel']['value']
                            elif 'value' in binding:
                                return binding['value']['value']
                        else:
                            resultList = []
                            for binding in data['results']['bindings']:
                                if 'valueLabel' in binding and binding['valueLabel']['value']:
                                     resultList.append(binding['valueLabel']['value'])
                                elif 'value' in binding:
                                    resultList.append(binding['value']['value'])
                            return ",".join(resultList)
                    return None
                else:
                    print(f"Failed to fetch {entity_code} {relation_code}, status code: {response.status}")
                    retries += 1
                    if retries < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Error fetching {entity_code} {relation_code}: {e}")
            retries += 1
            if retries < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
    return None


async def getAnswerOfRelation(json_data):
    updated_entities = {}
    async with ClientSession() as session:
        tasks = []
        for entity_code, entity in json_data.items():
            for relation in entity['pointed_relations']:
                task = asyncio.create_task(
                    fetch_relation_value(session, entity_code, relation['id'], pointing=False)
                )
                tasks.append((entity_code, relation, task))
            for relation in entity['pointing_relations']:
                task = asyncio.create_task(
                    fetch_relation_value(session, entity_code, relation['id'], pointing=True)
                )
                tasks.append((entity_code, relation, task))

        for entity_code, relation, task in tasks:
            relation_value = await task
            if relation_value:
                relation.update({'value': relation_value})
            if entity_code not in updated_entities:
                updated_entities[entity_code] = json_data[entity_code]

    return updated_entities


def getRelationValue(json_data):
    return asyncio.run(getAnswerOfRelation(json_data))


# 加载prop用于查询
def load_property_data(file_path):
    """
    加载属性数据文件，将 JSON 文件读取为列表并转换为字典，便于查找
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        property_list = json.load(f)
    # 将列表转换为字典，key 为 "id"，value 为该条目
    return {item['id']: item for item in property_list}

# 获取实体指向的属性/关系 或 指向实体的关系
def getAllRelationOfQID(QID, property_data):
    # 1. 构造查询：获取 QID 所指向的关系（QID -> 关系）
    sparql_query_pointed = f"""
    SELECT DISTINCT ?property ?propertyLabel WHERE {{
      wd:{QID} ?property ?target.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    Limit 100
    """

    # 2. 构造查询：获取指向 QID 的关系（关系 -> QID）
    sparql_query_pointing = f"""
    SELECT DISTINCT ?property ?propertyLabel WHERE {{
      ?item ?property wd:{QID}.
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 100 OFFSET 0
    """

    # 3. 设置请求头和请求体
    url = "https://query.wikidata.org/sparql"
    params = {
        'format': 'json'
    }

    # 4. 发送请求获取 QID 所指向的关系（QID -> 关系）
    params['query'] = sparql_query_pointed

    try:
        response_pointed = safe_request(url, params, proxies)
    except Exception as e:
        print(e)
        response_pointed=[]

    # 5. 发送请求获取指向 QID 的关系（关系 -> QID）
    params['query'] = sparql_query_pointing
    try:
        response_pointing =  safe_request(url, params, proxies)
    except Exception as e:
        print(e)
        response_pointing=[]
    pointed_relations = {}
    if response_pointed:
        # 6. 处理 QID 所指向的关系
        data_pointed = response_pointed

        for item in data_pointed['results']['bindings']:
            property_uri = item['property']['value']
            # 过滤，只保留以 "http://www.wikidata.org/prop/P" 开头的属性
            if 'wikidata' in property_uri:
                # 提取以 P 开头的属性 ID（如 P7033）
                property_id = property_uri.split('/')[-1]
                # 如果该 PID 在文件中查到，保存其 label
                if property_id in property_data:
                    pointed_relations[property_id] = property_data[property_id]["label"]
    pointing_relations = {}
    if response_pointing:
        # 7. 处理指向 QID 的关系
        data_pointing = response_pointing

        for item in data_pointing['results']['bindings']:
            property_uri = item['property']['value']
            # 过滤，只保留以 "http://www.wikidata.org/prop/P" 开头的属性
            if 'wikidata' in property_uri:
                # 提取以 P 开头的属性 ID（如 P7033）
                property_id = property_uri.split('/')[-1]
                # 如果该 PID 在文件中查到，保存其 label
                if property_id in property_data:
                    pointing_relations[property_id] = property_data[property_id]["label"]

    # 8. 返回结果，包含 ID 和 label
    return {
        "pointing_relations": pointing_relations,  # 返回字典形式，包含ID和label
        "pointed_relations": pointed_relations  # 返回字典形式，包含ID和label
    }
def get_wikipedia_title_from_qid(qid, language='en'):
    """
    根据Wikidata的QID获取对应语言的Wikipedia条目标题.

    参数:
    qid (str): Wikidata实体编号 (例如 'Q42').
    language (str): Wikipedia语言代码 (默认为 'en'，即英文).

    返回:
    str: Wikipedia条目标题 (如果找到).
    None: 如果未找到条目.
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        'action': 'wbgetentities',
        'ids': qid,
        'sites': f'{language}wiki',
        'props': 'sitelinks',
        'format': 'json'
    }

    try:
        data = safe_request(url, params, proxies) # 检查请求是否成功

        # 提取Wikipedia标题
        if 'entities' in data and qid in data['entities']:
            sitelinks = data['entities'][qid].get('sitelinks', {})
            wiki_key = f"{language}wiki"
            if wiki_key in sitelinks:
                return sitelinks[wiki_key]['title']
    except requests.RequestException as e:
        print(f"请求失败: {e}")

    return None
def relationLinking(entityIDs,question,itemInfo,myInfoBox,top_k,logicTree):
    # 创建一个字典来存储每个实体以及其关系信息
    InfoByEntity = {}
    retrieve_relation_List=set()
    # 遍历每一个实体ID
    for QId in entityIDs:
        if not QId:
            print(f"QId is None:{QId}")
            continue
        # 初始化当前实体的关系信息字典和关系标签集合
        if QId not in InfoByEntity:
            InfoByEntity[QId] = {}
        # 获取当前实体的所有关系项
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Construct the absolute path to wikidata_props.json, which is
        # in the SAME directory as wikidataUtills.py
        props_file_path = os.path.join(current_dir, 'wikidata_props.json')
        relaJson = getAllRelationOfQID(QId, load_property_data(props_file_path))
        if relaJson != {}:
            pointing_relations_pid = []
            pointed_relations_pid = []
            for item in relaJson["pointing_relations"]:
                pointing_relations_pid.append(relaJson["pointing_relations"][item])
            for item in relaJson["pointed_relations"]:
                pointed_relations_pid.append(relaJson["pointed_relations"][item])

            # 当前实体指向关系
            prompt_get_top6_rela = f"""
            现在请根据info，从两个后续列表中分别选出{top_k}个最可能与之相关的关系，并且最相关的关系排在最前面。
                info:{question}
                pointed_relations:{relaJson["pointed_relations"]}
                pointing_relations:{relaJson["pointing_relations"]}
            请注意你只需输出关系的id即可，请不要输出其他内容。
            参考输出json格式为：
            {{
                "pointed_relations":["", "", ""],
                "pointing_relations":["", "", ""]
            }}
            Please just output json format content, do not output any analysis text.
            """
            top6_entities,tokenCount = getModelResponse(prompt_get_top6_rela, "Now begin output：")
            logicTree.tokenCount+=tokenCount
            top6_entities = top6_entities.replace("```json", "").replace("```", "")
            json_data = json.loads(top6_entities)

            pointed_relations = []
            pointing_relations = []

            for item in json_data["pointed_relations"]:
                if item in relaJson["pointing_relations"]:
                    pointed_relations.append({"id":item, "label":relaJson["pointing_relations"][item]})
                if item in relaJson["pointed_relations"]:
                    pointed_relations.append({"id":item, "label":relaJson["pointed_relations"][item]})
                retrieve_relation_List.add(item)
            for item in json_data["pointing_relations"]:
                if item in relaJson["pointing_relations"]:
                    pointing_relations.append({"id": item, "label": relaJson["pointing_relations"][item]})
                if item in relaJson["pointed_relations"]:
                    pointing_relations.append({"id": item, "label": relaJson["pointed_relations"][item]})
                retrieve_relation_List.add(item)
            InfoByEntity[QId]['label'] = itemInfo[QId]['text']
            InfoByEntity[QId]['definition'] = itemInfo[QId]['definition']
            InfoByEntity[QId]['pointed_relations'] = pointed_relations
            InfoByEntity[QId]['pointing_relations'] = pointing_relations
            print(f"Linking {len(pointed_relations)+len(pointing_relations)} relations!")
    try:
        answersInfo = getRelationValue(InfoByEntity)
        #print(f"Entity Linking result：{json.dumps(answersInfo,indent=4)}")
    except Exception as e:
        answersInfo = InfoByEntity
        print(e)
    infos={
        "pointed": [],
        "pointing": []
    }
    for item in answersInfo:
        entityName = itemInfo[item]['text']
        for pointed_relation in answersInfo[item]["pointed_relations"]:
            if "value" in pointed_relation and "label" in pointed_relation:
                triple={
                    "head":entityName,
                    "relation":pointed_relation["label"],
                    "tail":pointed_relation["value"]
                }
                infos["pointed"].append(triple)
        for pointing_relation in answersInfo[item]["pointing_relations"]:
            if "value" in pointing_relation and "label" in pointing_relation:
                triple = {
                    "head": pointing_relation["value"],
                    "relation": pointing_relation["label"],
                    "tail": entityName
                }
                infos["pointed"].append(triple)
        infos["pointing"]=answersInfo[item]["pointing_relations"]
    myInfoBox.addGraph(json_array=infos)
    return retrieve_relation_List

if __name__ == '__main__':
    entityIDS=['Q525725']
    item_Info = {'Q525725': {'text': 'Andy Fickman', 'wikidata': 'Q525725', 'definition': 'Andy Fickman is an American film director, film producer, screenwriter, television director, television producer, and theatre director. His credits as a theater director include the premiere of the Reefer Madness! musical, the first Los Angeles production of the play Jewtopia, and the Los Angeles, Off-Broadway and London productions of Heathers: The Musical. He made his screen directing debut in 2002 with the teen sex'}}
    query = "What is Andy Fickman do?"
    top_k=3
    testInfoBox = infoBox()
    relationLinking(entityIDS,query,item_Info,testInfoBox,top_k)
    print(testInfoBox.graphInfo)