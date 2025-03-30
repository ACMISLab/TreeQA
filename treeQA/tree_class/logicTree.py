import json

from LLMs.models import getModelResponse
from treeQA.getQueryInfo import getQueryInfo

from treeQA.tree_class.infoBox import infoBox




class LogicTree:

    def __init__(self, data):
        self.data = data
        self.root = self.data["logic_tree"]
        self.fix_count = 0
        self.tokenCount = 0
    @staticmethod
    def logic_tree_init(query):
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
        result,tokenCount = getModelResponse(prompt, query)
        result=result.replace("```json", "").replace("```", "")
        print("Tree construction complete!")
        # 找到 JSON 字符串的起始和结束位置
        start_index = result.find('{')
        end_index = result.rfind('}') + 1

        # 提取 JSON 字符串
        json_string = result[start_index:end_index]

        # 解析 JSON 字符串
        try:
            data = json.loads(json_string)
            data["input_question"] = query
            return data,tokenCount
        except json.JSONDecodeError as e:
            print("error" + json_string)
            print(f"Decomposition failed, invalid JSON format.{e}")
        return None

    def traverse(self):
        qa_pairs = []

        def _traverse(node):
            if "sub_question" in node and "hypothesis_answer" in node:
                qa_pairs.append((node["sub_question"], node["hypothesis_answer"]))
            if "children" in node:
                for child in node["children"]:
                    _traverse(child)

        _traverse(self.root)
        return qa_pairs

    def update_node(self, path, new_value):
        node = self.root
        for index in path:
            node = node["children"][index]

        node.update(new_value)

    def get_node_by_path(self, path):
        node = self.root
        for index in path:
            node = node["children"][index]
        return node

    def refine_subtree(self, path):
        current_node = self.get_node_by_path(path)
        question = self.data["input_question"]

        def factCheck(current_node,question):
            print(f"#####################Begin self-adaptive reasoning!#####################")
            checkInfoBox = infoBox()
            childQuestion = current_node.get("sub_question")
            hypothesis_answer = current_node.get("hypothesis_answer")
            # 从问题出发获取相关信息
            getQueryInfo(childQuestion+hypothesis_answer, checkInfoBox, self)

            # 判断答案是否有误,并填充引用来源
            factCheckPrompt = f"""Please verify whether the answer is correct based on the given Info.  

- If no relevant Info is provided (e.g., ["No Information provided."] or []), set `"isTrue": "unknown"` and `"fact_sufficient": false`.  
- If the answer is correct and no reason is needed, set `"isTrue": true`.  
- If the answer is incorrect, set `"isTrue": false`, provide the reason for the error, and include the correct answer in the reason.  
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
            result,tokenCount = getModelResponse(factCheckPrompt, f"question：{childQuestion}\nanswer:{hypothesis_answer}\nInfo：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}")
            result=result.replace('```json', '').replace('```', '')
            self.tokenCount+=tokenCount
            # 找到 JSON 字符串的起始和结束位置
            start_index = result.find('{')
            end_index = result.rfind('}') + 1

            # 提取 JSON 字符串
            json_string = result[start_index:end_index]

            # 解析 JSON 字符串

            resultJson = json.loads(json_string)
            if resultJson["isTrue"] == "unknown" and resultJson["fact_sufficient"]==False:

                prompt_new_cue="""The available information is insufficient to answer the question. Based on the given information and the question, generate a new clue to help retrieve the missing information needed to answer it.  
                ### Output Format (JSON only):  
                ```json
                {
                    "new_clue": "<New clue>"
                }"""
                new_clue, tokenCount=getModelResponse(prompt_new_cue, f"question：{childQuestion}\nCurrent Info：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}")
                self.tokenCount += tokenCount
                print("#########No useful information obtained, new leads provided:#############"+new_clue)
                getQueryInfo(new_clue, checkInfoBox,self)
                result, tokenCount = getModelResponse(factCheckPrompt,
                                          f"question：{childQuestion}\nanswer:{hypothesis_answer}\nInfo：\t\ntextInfo:{checkInfoBox.textInfo}\t\ngraphInfo:{checkInfoBox.graphInfo}")
                result=result.replace('```json', '').replace('```', '')
                self.tokenCount += tokenCount
                # 找到 JSON 字符串的起始和结束位置
                start_index = result.find('{')
                end_index = result.rfind('}') + 1
                # 提取 JSON 字符串
                json_string = result[start_index:end_index]
                # 解析 JSON 字符串
                resultJson = json.loads(json_string)

            # 若无误进入下一步，添加相关参考信息
            if resultJson["isTrue"]:
                print("############Evidence support node!##############")
                # 这里向当前的节点添加reference信息
                if "ref" not in current_node:
                    current_node["ref"] = {}

                # 从 infoBox 中获取完整的维基百科文章
                wikipedia_ref_with_text = []
                if "wikipedia" in resultJson["ref"]:
                    for title in resultJson["ref"]["wikipedia"]:
                        for item in checkInfoBox.textInfo:
                            if item[0]['id'] == title:
                                wikipedia_ref_with_text.append(f"{title}||{item[0]['content']}")
                                break  # 找到匹配项后退出内层循环
                # 更新 ref 字段，将维基百科的引用替换为带有文本的引用
                current_node["ref"].update(resultJson["ref"])
                if wikipedia_ref_with_text:
                    current_node["ref"]["wikipedia"] = wikipedia_ref_with_text
                return current_node
            # 若有误将错误原因记下，修正整个过程
            if not resultJson["isTrue"]:
                errorReason = resultJson["reason"]
                print(f"############Conflict were found:###################\n\n{errorReason}")
                self.fix_count+=1
                fixHypothesisPropmt = f"""An error was found in the step {current_node} of the current assumption, with a specific error reason of {errorReason}.
                           Please fully review and refactor all sub_questions and hypothesis_answer starting at subtree node:{current_node}.
                           The refactoring process involves correcting or redoing each step as necessary based on the latest information and logic to ensure that the answer does not deviate from the current question:{question}. 
                   Keeping the original subtree depth and structure,you only need to fix the errors in the subtree node, and do not add any deeper child node in the subtree. 
                   Please just output json format content, do not output any analysis text.
                   """
                result, tokenCount = getModelResponse(fixHypothesisPropmt, f"Please be careful that current responses do not deviate from the question:{question}")
                result = result.replace('```json', '').replace('```', '')
                self.tokenCount += tokenCount
                print(f"################New subtree:######################\n{result}")

                # 找到 JSON 字符串的起始和结束位置
                start_index = result.find('{')
                end_index = result.rfind('}') + 1

                # 提取 JSON 字符串
                json_string = result[start_index:end_index]
                # 解析 JSON 字符串

                fixedHypothesis = json.loads(json_string)
                # 更新当前节点和其子节点
                self.update_node(path, fixedHypothesis)

                print(f"################Subtree update complete!##################")
                # 添加参考信息
                if "ref" not in current_node:
                    current_node["ref"] = {}

                # 从 infoBox 中获取完整的维基百科文章
                wikipedia_ref_with_text = []
                if "wikipedia" in resultJson["ref"]:
                    for title in resultJson["ref"]["wikipedia"]:
                        for item in checkInfoBox.textInfo:
                            if item[0]['id'] == title:
                                wikipedia_ref_with_text.append(f"{title}||{item[0]['content']}")
                                break  # 找到匹配项后退出内层循环

                # 更新 ref 字段，将维基百科的引用替换为带有文本的引用
                current_node["ref"].update(resultJson["ref"])
                if wikipedia_ref_with_text:
                    current_node["ref"]["wikipedia"] = wikipedia_ref_with_text
            else:
                print("Unable to check node, continue to next node.")
            return current_node

        updated_node = factCheck(current_node,question)
        return updated_node

    def check_and_refine(self):
        """
        遍历整个逻辑树，对每个节点进行 factCheck()，并根据需要进行 refineTree()。
        """

        def _recursive_check(node, path):
            if "sub_question" in node and "hypothesis_answer" in node:
                print(f"Checking:{path}")
                self.refine_subtree(path)

            if "children" in node:
                for i, child in enumerate(node["children"]):
                    _recursive_check(child, path + [i])

        _recursive_check(self.root, [])

    def print_tree(self, node=None, indent=0, output_lines=None):
        """
        将整棵树转换为 Markdown 文本，并打印出来。
        返回 Markdown 文本的所有行。
        """
        if output_lines is None:
            output_lines = []

        if node is None:
            node = self.root
            output_lines.append(f"**Input Question:** {self.data['input_question']}\n")
            output_lines.append(f"**Final Answer:** {self.data['answer']}\n")

        if "sub_question" in node and "hypothesis_answer" in node:
            output_lines.append("  " * indent + f"- Question: {node['sub_question']}")
            output_lines.append("  " * indent + f"  Answer: {node['hypothesis_answer']}")
            if "ref" in node:
                output_lines.append("  " * indent + "  References:")
                for ref_type, ref_list in node["ref"].items():
                    if ref_type == "wikipedia":
                        for ref in ref_list:
                            output_lines.append("  " * indent + f"    - Wikipedia: {ref}")
                    elif ref_type == "wikidata":
                        for ref in ref_list:
                            output_lines.append("  " * indent + f"    - Wikidata: {ref}")
                    else:
                        output_lines.append("  " * indent + f"    - {ref_type}: {ref_list}")

        if "children" in node:
            for child in node["children"]:
                self.print_tree(child, indent + 1, output_lines)

        return output_lines
    def update_final_answer(self):
        info=self.data["logic_tree"]
        prompt=f"""
            Now based all information and your own knowledge,please give a final answer to the question.
            question:{self.data["input_question"]}
            information:{info}
         """
        final_answer,tokenCount=getModelResponse(prompt,self.data["input_question"])
        self.tokenCount += tokenCount
        print(f"########the final answer is:####################\n{final_answer}")
        self.data["answer"]=final_answer

    def to_json(self):
        """
        将整棵树转换为 JSON 格式的字符串。

        Returns:
            一个表示整棵树的 JSON 字符串。
        """
        return json.dumps(self.data, indent=4),self.fix_count