import json
import concurrent.futures
import threading
from queue import Queue

# 假设这些是你项目中的模块
from LLMs.models import getModelResponse
from treeQA.getQueryInfo import getQueryInfo
from treeQA.tree_class.infoBox import infoBox

# 创建一个线程局部存储，用于保存每个线程的 tokenCount
# 这比在每次LLM调用后都加锁更高效
thread_local = threading.local()


class LogicTree:
    """
    一个表示和操作逻辑推理树的类。
    支持串行和层级并行的验证与修正机制。
    """

    def __init__(self, data):
        """
        初始化LogicTree实例。

        Args:
            data (dict): 包含初始逻辑树结构的字典。
        """
        self.data = data
        self.root = self.data.get("logic_tree", {})
        self.fix_count = 0
        self.tokenCount = 0
        # 使用锁来保护对共享资源（如计数器和树结构）的并发访问
        self.lock = threading.Lock()

    @staticmethod
    def logic_tree_init(query):
        """
        静态方法，用于从一个问题（query）初始化一个逻辑树。
        它调用LLM来分解问题并生成初始的树结构。
        """
        prompt = f"""You are an intelligent assistant who is good at analyzing and reasoning, and your task is to construct a logic tree to break down and reason step by step according to the complex questions posed by the user, and finally arrive at the answer.

    **Rules for constructing a logic tree:**

    1. **Root:** the user's main Question.
    2. **Sub-nodes:** Each sub-node contains a **Sub-question** and a **Specific Hypothesis Answer** based on your own knowledge.
        * **Sub-questions:** are used to guide reasoning and information retrieval, and each sub-question should address only a single entity or relationship.
        * **Concrete Hypothesis Answer:** is a **concrete answer** to a sub-question given by your own knowledge, rather than an abstract or broad hypothesis.This answer should be the one that the model considers most likely, even though it may not be correct,and the answer should also be richly detailed.
    3. **Construction principles:** The construction of the logic tree follows the pyramid principle:
        * **Conclusion first:** The hypothetical answer of a parent node should be a generalization of the information of its children.
        * **Correspondence between top and bottom:** The information of a child node should support the hypothetical answer of the parent node.
        * **MECE Principle: ** Child nodes at the same level should be independent of each other and try to cover all possibilities.
        * **Logical progression:** The generation of child nodes should follow a certain logical order (e.g., time, cause and effect, etc.).

    * **Reasoning steps:**

    1. **Question Decomposition & Hypothesis Generation:** Based on the information (question or statement) in the current node, propose a simpler sub-question and give a **specific hypothetical answer** based on your own knowledge base.
    2. **Answer Synthesis:** When the logic tree is constructed, synthesize and reason out the final answer based on the information from all nodes.

    **Output Format:**

    Please strictly follow the following JSON format for output:

    ```json
    {{
      "input_question":"<main_question>",
      "logic_tree": {{
          "children": [
            {{
              "sub_question":"<subquestion1>",
              "hypothesis_answer":"<specific hypothesis answer 1>",
              "children": [
                {{
                  "sub_question":"<sub-sub-question 1.1>",
                  "hypothesis_answer": "<specific hypothetical answer 1.1>",
                  "children": [
                    {{
                      "sub_question":"<children_subquestion 1.1.1>",
                      "hypothesis_answer":"<specific hypothetical answer 1.1.1>"
                    }}
                  ]
                }}
              ]
            }}
          ]
        }},
      "answer": "<final answer>"
    }}
    ```

    Caveats:


    Each hypothetical answer must be a concrete, verifiable answer, not an abstract hypothesis.

    Ensure that the logic tree construction process follows the pyramid principle.

    The output should be concise and intuitive, with the lowest level question or hypothesis containing only the least granular information.

    Please just output json format content, do not output any analysis.

    Now the user question is:{query}
    """
        result, tokenCount = getModelResponse(prompt, query)
        result = result.replace("```json", "").replace("```", "")
        print("Tree construction complete!")

        try:
            json_string = result[result.find('{'):result.rfind('}') + 1]
            data = json.loads(json_string)
            data["input_question"] = query
            return data, tokenCount
        except json.JSONDecodeError as e:
            print(f"Error: Decomposition failed, invalid JSON format. {e}")
            print("Received string:", result)
            return None, tokenCount

    def get_node_by_path(self, path):
        """通过路径获取树中的一个节点。"""
        node = self.root
        try:
            for index in path:
                node = node["children"][index]
            return node
        except (KeyError, IndexError):
            return None

    def _increment_token_count(self, count):
        """线程安全地增加全局 token 计数。"""
        with self.lock:
            self.tokenCount += count

    def _increment_fix_count(self):
        """线程安全地增加全局 fix 计数。"""
        with self.lock:
            self.fix_count += 1

    def refine_subtree(self, path):
        """
        对指定路径的节点进行验证和修正。此方法是线程安全的。
        它直接修改树的节点，不返回值。
        """
        # 初始化当前线程的 token 计数器
        thread_local.token_count = 0

        current_node = self.get_node_by_path(path)
        if not current_node:
            print(f"Warning: Node at path {path} not found, possibly due to a concurrent modification.")
            return

        question = self.data["input_question"]

        print(f"##################### Begin self-adaptive reasoning for path {path} #####################")
        checkInfoBox = infoBox()
        childQuestion = current_node.get("sub_question")
        hypothesis_answer = current_node.get("hypothesis_answer")

        getQueryInfo(childQuestion + hypothesis_answer, checkInfoBox, self)

        factCheckPrompt = f"""Please verify whether the answer is correct based on the given Info.  

    - If no relevant Info is provided (e.g., ["No Information provided."] or []), set `"isTrue": "unknown"` and `"fact_sufficient": false`.  
    - If the answer is correct and no reason is needed, set `"isTrue": true`.  
    - If the answer is incorrect, set `"isTrue": false`, provide the reason for the error, and please also suggest possible correct answers in the reason.  
    - Always include reference information (`ref`) when available, for both correct and incorrect answers.  

    ### Reference Formatting Rules:  
    - If no Info is available or unrelated to the question, set `"fact_sufficient": false`, `"ref": "No Information provided."`, and `"isTrue": "unknown"`.  
    - If citing `textInfo`, provide only the Wikipedia article's title ID (omit full text).  
    - If citing `graphInfo`, provide up to 3 relevant triplets in the format: `entityLabel-relationLabel-Value` from Wikidata.  
    - Do not fabricate information—references must match the provided Info.  

    ### Output Format (JSON only):  
    ```json
    {{
        "isTrue": true/false/unknown,
        "fact_sufficient": true/false,
        "reason": "<None>/<reason>",
        "ref": {{
            "wikipedia": ["<textInfo id>"],
            "wikidata": ["entityLabel-relationLabel-Value"]
        }}
    }}
    ```
            """
        query_info = f"question：{childQuestion}\nanswer:{hypothesis_answer}\nInfo：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}"
        result, tokenCount = getModelResponse(factCheckPrompt, query_info)
        result = result.replace('```json', '').replace('```', '')
        thread_local.token_count += tokenCount

        try:
            json_string = result[result.find('{'):result.rfind('}') + 1]
            resultJson = json.loads(json_string)
        except json.JSONDecodeError:
            print(f"Error: Failed to decode JSON from fact-check LLM for path {path}.")
            self._increment_token_count(thread_local.token_count)
            return

        if resultJson.get("isTrue") == "unknown" and not resultJson.get("fact_sufficient"):
            prompt_new_cue = """The available information is insufficient to answer the question. Based on the given information and the question, generate a new clue to help retrieve the missing information needed to answer it.  
                ### Output Format (JSON only):  
                ```json
                {
                    "new_clue": "<New clue>"
                }"""
            new_clue_query = f"question：{childQuestion}\nCurrent Info：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}"
            new_clue_str, tokenCount = getModelResponse(prompt_new_cue, new_clue_query)
            thread_local.token_count += tokenCount
            print(f"######### No useful information for path {path}, new clue provided: {new_clue_str} #############")

            getQueryInfo(new_clue_str, checkInfoBox, self)

            query_info_updated = f"question：{childQuestion}\nanswer:{hypothesis_answer}\nInfo：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}"
            result, tokenCount = getModelResponse(factCheckPrompt, query_info_updated)
            result = result.replace('```json', '').replace('```', '')
            thread_local.token_count += tokenCount
            try:
                json_string = result[result.find('{'):result.rfind('}') + 1]
                resultJson = json.loads(json_string)
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON from second fact-check for path {path}.")
                self._increment_token_count(thread_local.token_count)
                return

        if resultJson.get("isTrue") is True:
            print(f"############ Evidence supports node at path {path}! ##############")
            # 【修改点】调用辅助函数添加详细引用
            self._add_references_to_node(current_node, resultJson, checkInfoBox)

        elif resultJson.get("isTrue") is False:
            errorReason = resultJson.get("reason", "No reason provided.")
            print(f"############ Conflict found for node at path {path}: {errorReason} ###################")
            self._increment_fix_count()

            fixHypothesisPropmt = f"""An error was found in the step {current_node} of the current assumption, with a specific error reason of {errorReason}.
                           Please fully review and refactor all sub_questions and hypothesis_answer starting at subtree node:{current_node}.
                           The refactoring process involves correcting or redoing each step as necessary based on the latest information and logic to ensure that the answer does not deviate from the current question:{question}. 
                   Keeping the original subtree depth and structure,you only need to fix the errors in the subtree node, and do not add any deeper child node in the subtree. 
                   Please just output json format content, do not output any analysis text.
                   """
            fix_query = f"Please be careful that current responses do not deviate from the question:{question}"
            result, tokenCount = getModelResponse(fixHypothesisPropmt, fix_query)
            result = result.replace('```json', '').replace('```', '')
            thread_local.token_count += tokenCount
            print(f"################ New subtree for path {path}: {result} ################")

            try:
                json_string = result[result.find('{'):result.rfind('}') + 1]
                fixedHypothesis = json.loads(json_string)

                # 线程安全地更新节点
                with self.lock:
                    current_node.clear()
                    current_node.update(fixedHypothesis)

                # 【修改点】在修正后，也调用辅助函数添加详细引用
                self._add_references_to_node(current_node, resultJson, checkInfoBox)

                print(f"################ Subtree at path {path} update complete! ################")
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON for subtree fix at path {path}.")
        else:
            print(f"Unable to check node at path {path}, continue to next node.")

        # 在线程结束时，将该线程的 token 总数加到全局计数器
        self._increment_token_count(thread_local.token_count)

    def _add_references_to_node(self, node, ref_json, info_box):
        """
        【新增辅助函数】线程安全地向节点添加引用信息，包括详细的维基百科文本。
        """
        with self.lock:
            if "ref" not in node:
                node["ref"] = {}

            # 提取维基百科ID
            wiki_ids = ref_json.get("ref", {}).get("wikipedia", [])

            # 查找并构建详细的维基百科引用
            wikipedia_ref_with_text = []
            if wiki_ids:
                for title_id in wiki_ids:
                    found = False
                    for item in info_box.textInfo:
                        # 假设 item[0]['id'] 是文章标题ID
                        if item[0]['id'] == title_id:
                            content = item[0]['content']
                            wikipedia_ref_with_text.append(f"{title_id}||{content}")
                            found = True
                            break
                    if not found:
                        # 如果在info_box中没找到，就只添加ID
                        wikipedia_ref_with_text.append(title_id)

            # 更新节点的引用信息
            # 先复制一份，避免直接修改传入的ref_json
            updated_ref = ref_json.get("ref", {}).copy()
            if wikipedia_ref_with_text:
                updated_ref["wikipedia"] = wikipedia_ref_with_text

            # 合并到节点的ref字段
            # 这里使用合并，而不是直接替换，以保留可能已有的其他引用类型
            for key, value in updated_ref.items():
                if key not in node["ref"]:
                    node["ref"][key] = []
                # 避免重复添加
                for item in value:
                    if item not in node["ref"][key]:
                        node["ref"][key].append(item)
    def check_and_refine_parallel(self, max_workers=5):
        """
        【并行方法】使用广度优先和线程池并行地检查和修正树。
        这是推荐的、用于降低延迟的方法。
        """
        if not self.root.get("children"):
            print("Tree has no children to check.")
            return

        q = Queue()
        for i, child in enumerate(self.root.get("children", [])):
            q.put((child, [i]))

        level = 1
        while not q.empty():
            level_size = q.qsize()
            if level_size == 0:
                break

            print(f"\n--- Processing Level {level} with {level_size} nodes in parallel ---")

            tasks_for_current_level = []
            paths_for_next_level = []
            for _ in range(level_size):
                node, path = q.get()
                if "sub_question" in node and "hypothesis_answer" in node:
                    tasks_for_current_level.append(path)
                paths_for_next_level.append(path)

            if tasks_for_current_level:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    executor.map(self.refine_subtree, tasks_for_current_level)

            print(f"--- Level {level} processing complete ---")

            # 重新收集下一层的节点，因为树的结构可能已经改变
            for path in paths_for_next_level:
                parent_node = self.get_node_by_path(path)
                if parent_node and "children" in parent_node:
                    for i, child in enumerate(parent_node["children"]):
                        new_path = path + [i]
                        q.put((child, new_path))

            level += 1

    def check_and_refine_serial(self):
        """
        【串行方法】使用深度优先递归的方式检查和修正树。
        用于功能验证或性能对比。
        """

        def _recursive_check(node, path):
            if "sub_question" in node and "hypothesis_answer" in node:
                print(f"Checking (serial): {path}")
                self.refine_subtree(path)

            if "children" in node:
                # 必须重新获取节点，因为它可能已被refine_subtree修改
                current_node = self.get_node_by_path(path)
                if current_node and "children" in current_node:
                    for i, child in enumerate(current_node["children"]):
                        _recursive_check(child, path + [i])

        _recursive_check(self.root, [])

    def update_final_answer(self):
        """调用LLM，基于修正后的整棵树信息生成最终答案。"""
        info = self.data["logic_tree"]
        prompt = f"""
            Now based all information from the verified logic tree and your own knowledge, please give a final, concise answer to the question.
            question: {self.data["input_question"]}
            verified logic tree information: {info}
         """
        final_answer, tokenCount = getModelResponse(prompt, self.data["input_question"])
        self.tokenCount += tokenCount
        print(f"######## The final answer is: ####################\n{final_answer}")
        self.data["answer"] = final_answer

    def to_json(self):
        """将整棵树（包括最终答案）转换为格式化的JSON字符串。"""
        return json.dumps(self.data, indent=4), self.fix_count

    def print_tree(self, node=None, indent=0, output_lines=None):
        """将整棵树转换为易于阅读的Markdown格式文本。"""
        if output_lines is None:
            output_lines = []

        if node is None:
            node = self.root
            output_lines.append(f"**Input Question:** {self.data.get('input_question', 'N/A')}\n")
            output_lines.append(f"**Final Answer:** {self.data.get('answer', 'N/A')}\n")

        if "sub_question" in node and "hypothesis_answer" in node:
            prefix = "  " * indent
            output_lines.append(f"{prefix}- Question: {node['sub_question']}")
            output_lines.append(f"{prefix}  Answer: {node['hypothesis_answer']}")
            if "ref" in node and node["ref"]:
                output_lines.append(f"{prefix}  References:")
                for ref_type, ref_list in node["ref"].items():
                    if ref_list:
                        for ref in ref_list:
                            output_lines.append(f"{prefix}    - {ref_type.capitalize()}: {ref}")

        if "children" in node:
            for child in node["children"]:
                self.print_tree(child, indent + 1, output_lines)

        return "\n".join(output_lines)


# ================== Example Usage ==================
if __name__ == '__main__':
    # 这是一个模拟的运行流程
    # 1. 定义你的依赖项（这里用mock函数代替）
    def getModelResponse(prompt, query):
        print("--- Mock LLM Call ---")
        print(f"Prompt: {prompt[:100]}...")
        print(f"Query: {query}")
        # 模拟LLM返回的JSON
        if "construct a logic tree" in prompt:
            mock_tree = {
                "input_question": "Who was the President of the United States when humans made their third landing on the moon?",
                "logic_tree": {
                    "children": [
                        {
                            "sub_question": "When was the third manned moon landing?",
                            "hypothesis_answer": "The third manned moon landing was Apollo 12 in November 1969.",
                            "children": [
                                {
                                    "sub_question": "Who was the US President in November 1969?",
                                    "hypothesis_answer": "Lyndon B. Johnson was the US President in November 1969."
                                }
                            ]
                        },
                        {
                            "sub_question": "Which mission was the third moon landing?",
                            "hypothesis_answer": "Apollo 14 was the third mission to land on the moon.",
                            "children": [
                                {
                                    "sub_question": "When did Apollo 14 land on the moon?",
                                    "hypothesis_answer": "Apollo 14 landed on February 5, 1971."
                                }
                            ]
                        }
                    ]
                },
                "answer": "Initial answer placeholder"
            }
            return json.dumps(mock_tree), 500
        elif "verify whether the answer is correct" in prompt:
            mock_verification = {"isTrue": False, "fact_sufficient": True,
                                 "reason": "The third landing was Apollo 14 in 1971, not Apollo 12. The president in 1971 was Richard Nixon, not LBJ.",
                                 "ref": {"wikipedia": ["Apollo_14"],
                                         "wikidata": ["Richard_Nixon-president_of-United_States"]}}
            return json.dumps(mock_verification), 100
        elif "refactor all sub_questions" in prompt:
            mock_fix = {
                "sub_question": "When was the third manned moon landing?",
                "hypothesis_answer": "The third manned moon landing was Apollo 14 on February 5, 1971.",
                "children": [
                    {
                        "sub_question": "Who was the US President on February 5, 1971?",
                        "hypothesis_answer": "Richard Nixon was the US President on February 5, 1971."
                    }
                ]
            }
            return json.dumps(mock_fix), 200
        elif "give a final, concise answer" in prompt:
            return "Richard Nixon.", 50
        return "{}", 10


    def getQueryInfo(query, infoBox, tree_instance):
        print(f"--- Mock Retrieval for query: '{query}' ---")
        infoBox.textInfo = [{"id": "Apollo_14",
                             "content": "Apollo 14 was the third United States Apollo mission to land on the Moon..."}]
        infoBox.graphInfo = ["Richard_Nixon-position_held-President_of_the_United_States"]


    class infoBox:
        def __init__(self):
            self.textInfo = []
            self.graphInfo = []


    # 2. 运行主流程
    user_query = "Who was the President of the United States when humans made their third landing on the moon?"

    # 初始化树
    initial_data, init_tokens = LogicTree.logic_tree_init(user_query)
    if initial_data:
        my_tree = LogicTree(initial_data)
        my_tree.tokenCount += init_tokens

        print("\n=============== INITIAL TREE ===============\n")
        print(my_tree.print_tree())

        # 使用并行方法进行验证和修正
        print("\n=============== STARTING PARALLEL REFINEMENT ===============\n")
        my_tree.check_and_refine_parallel(max_workers=2)  # 使用2个worker进行并行处理

        print("\n=============== REFINED TREE ===============\n")
        print(my_tree.print_tree())

        # 生成最终答案
        my_tree.update_final_answer()

        # 输出最终结果
        final_json, fix_count = my_tree.to_json()
        print("\n=============== FINAL JSON OUTPUT ===============\n")
        print(final_json)
        print(f"\nTotal fixes: {fix_count}")
        print(f"Total tokens used: {my_tree.tokenCount}")
