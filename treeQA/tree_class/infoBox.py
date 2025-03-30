class infoBox:
    def __init__(self):
        # 初始化两个列表属性
        self.graphInfo = []
        self.textInfo = []
        self.sparqlInfo = []
        # 使用集合来存储已添加的信息的唯一标识符
        self.seen_graphs = set()
        self.seen_texts = set()

    def addGraph(self, json_array):
        for items in json_array:
            # 遍历新的信息并添加到列表中
            jsonItems = json_array[items]
            for item in jsonItems:
                key = (item.get('head', ''), item.get('relation', ''), item.get('tail'))
                if key not in self.seen_graphs:
                    self.graphInfo.append([item])  # 确保添加的是一个嵌套列表
                    self.seen_graphs.add(key)

    def addText(self, new_info):
        # 遍历新的信息并添加到列表中
        if new_info:
            for json_array in new_info:
                key = json_array.get('id', '')
                if key and key not in self.seen_texts:
                    self.textInfo.append([json_array])
                    self.seen_texts.add(key)
