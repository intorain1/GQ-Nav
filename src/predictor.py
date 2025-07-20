from graph import OptimizedTimeObjectGraph
from openai import OpenAI
import os
import http.client
import json
import sys
import random

class Predictor:
    def __init__(self):
        # skill和object闭集配置
        self.skillspace = ["move", "go_upstairs", "go_downstairs", "open_door"]
        self.nodespace = ["chair", "sofa", "door", "window", "table", "refrigerator", "vase", "tv", "sink", "bed", "oven", "shower", "mirror", "cloth", "plant"]
        # llm配置
        self.model = "gpt-3.5-turbo"
        self.temperature = 0.0
        self.api_key = "sk-Rlk4nftf04HJqggDupZ7uM4Ur7TNUIHhlAlStDI2hCQtTLc5"
        self.url = "api.chatanywhere.tech"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.client = OpenAI(api_key = self.api_key, base_url="https://api.chatanywhere.tech/v1")
        
        self.detected_graph = OptimizedTimeObjectGraph()
        self.predicted_graph = OptimizedTimeObjectGraph()
        self.graph_prompt = None

        self.current_position = None  # Current position in the scene map
        
        self.instruction = None
        self.action_chain=[] #用物体进行描述
        self.imagined_graph_chain = []
        
        self.system_prompt = '''
You are a scene-map-aware navigation agent.

Your skillset allows you to:
1. Read local scene maps composed of objects. Understand the scene by identifying objects and their relationships.
2. Plan safe, low-cost paths to reach a target position using the scene map.
3. Select and apply real-world navigation skills to move toward the target.
4. Predicted and update possible surrounding objects every time you move to a new position.

### Scene Map Format
The scene map is an ordered list of object-groups.
- Each object-group is presented by a set of objects that can be detected by the agent from current view.  
- The format of an object-group is:
['chair', 'sofa', 'door', 'window', 'table']
- Each object is represented by its caption (e.g., "chair", "sofa"). 
- The objects in an object-group are unordered, indicating only that they can be observed from the same viewpoint.

- The format of a scene map is:
{
    ['chair', 'sofa', 'door', 'window', 'table'],
    ['sofa', 'refrigerator', 'door', 'table', 'vase'],
    ...
}
- The object-groups in the scene map are ordered, indicating the sequence of views as the agent moves through the environment.


### Your Task
Given the current scene map, the target instruction, and your current position, your job is to:
- Generate the next **5 actions in order** , leading the robot to gradually approach the target.
- Each action must specify the direction, the navigation skill used, and the the target object reached.

**Solid Objects:** You are only allowed to choose solid objects as the "target objects" such as furniture, doors, windows, stairs, and walls.
**Rules for Skill Selection**
- A skill is chosen from the set: ["move", "go_upstairs", "go_downstairs",  "open_door"].
- Use **go_upstairs** or **go_downstairs** when moving through stairs (real or imagined).
- Use **open_door** when passing through a door (real or imagined).
- Use **move** for all other standard movements.

### Output Format
Each action must be expressed as:
{"action": 
{
    "num_of_order": <1-5>,
    "skill": <one of [move, go_upstairs, go_downstairs, open_door]>,
    "object": <name of the target object reached>
}
}

After each action, **imagine a object-group** you can see after reaching the taget position according to the instruction and the current scene map.
Prediction of each action must be expressed as an object-group:
{"imagined_view": ['bed', 'oven', 'door', 'sink']}

Notice that the scene map is incomplete and may not contain all the objects in the environment so that you can not always choose existing objects as the target position.
You must create at least three new object in each predicted object-group that does not exist in the given scene map. These objects should be plausible based on the current instruction and daily life knowledge.


### Final Output Format
Your response must be a **list alternating between actions and their predicted scene maps**, with no other text or commentary:

Example:(Assume door is not included in the given scene map, but you can infer it from the instruction)
{"action": {"num_of_order":1, "skill":"open_door", "object":"door"}}
{"imagined_view": ['chair', 'window', 'sink']}
{"action": {"num_of_order":2, "skill":"move", "object":"sink"}}
{"imagined_view": ['door', 'shower', 'mirror', 'cloth']}
...
The objects in the imagined_view and action can only be chosen from:'''
        self.system_prompt += f'''{self.nodespace}\n'''
        self.system_prompt += f'''Do **not** include any other explanations or extra text in your output.
Remeber you must generate 5 actions and matched imagined_view in total! Think carefully of the objects you will see along the way!
'''

        self.user_prompt =""
        #优化方向：现在的前几个goal只由instruction决定，并不合理，应该在路径中途细化想象；
        #skill没有体现，是一个空壳参数
        #没加入想象错误的反馈
    
    def set_llm(self, model: str, url:str, api_key: str, temperature: float):
        self.model = model
        self.url = url
        self.api_key = api_key
        self.temperature = temperature
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.client = OpenAI(api_key=self.api_key, base_url=f"https://{self.url}/v1")

    def set_system_prompt(self):
        # 根据nodespace，需要微调prompt中的示例
        self.system_prompt = '''
You are a scene-map-aware navigation agent.

Your skillset allows you to:
1. Read local scene maps composed of objects. Understand the scene by identifying objects and their relationships.
2. Plan safe, low-cost paths to reach a target position using the scene map.
3. Select and apply real-world navigation skills to move toward the target.
4. Predicted and update possible surrounding objects every time you move to a new position.

### Scene Map Format
The scene map is an ordered list of object-groups.
- Each object-group is presented by a set of objects that can be detected by the agent from current view.  
- The format of an object-group is:
['chair', 'sofa', 'door', 'window', 'table']
- Each object is represented by its caption (e.g., "chair", "sofa"). 
- The objects in an object-group are unordered, indicating only that they can be observed from the same viewpoint.

- The format of a scene map is:
{
    ['chair', 'sofa', 'door', 'window', 'table'],
    ['sofa', 'refrigerator', 'door', 'table', 'vase'],
    ...
}
- The object-groups in the scene map are ordered, indicating the sequence of views as the agent moves through the environment.


### Your Task
Given the current scene map, the target instruction, and your current position, your job is to:
- Generate the next **5 actions in order** , leading the robot to gradually approach the target.
- Each action must specify the direction, the navigation skill used, and the the target object reached.

**Solid Objects:** You are only allowed to choose solid objects as the "target objects" such as furniture, doors, windows, stairs, and walls.
**Rules for Skill Selection**
- A skill is chosen from the set: ["move", "go_upstairs", "go_downstairs",  "open_door"].
- Use **go_upstairs** or **go_downstairs** when moving through stairs (real or imagined).
- Use **open_door** when passing through a door (real or imagined).
- Use **move** for all other standard movements.

### Output Format
Each action must be expressed as:
{"action": 
{
    "num_of_order": <1-5>,
    "skill": <one of [move, go_upstairs, go_downstairs, open_door]>,
    "object": <name of the target object reached>
}
}

After each action, **imagine a object-group** you can see after reaching the taget position according to the instruction and the current scene map.
Prediction of each action must be expressed as an object-group:
{"imagined_view": ['bed', 'oven', 'door', 'sink']}

Notice that the scene map is incomplete and may not contain all the objects in the environment so that you can not always choose existing objects as the target position.
You must create at least three new object in each predicted object-group that does not exist in the given scene map. These objects should be plausible based on the current instruction and daily life knowledge.


### Final Output Format
Your response must be a **list alternating between actions and their predicted scene maps**, with no other text or commentary:

Example:(Assume door is not included in the given scene map, but you can infer it from the instruction)
{"action": {"num_of_order":1, "skill":"open_door", "object":"door"}}
{"imagined_view": ['chair', 'window', 'sink']}
{"action": {"num_of_order":2, "skill":"move", "object":"sink"}}
{"imagined_view": ['door', 'shower', 'mirror', 'cloth']}
...
The objects in the imagined_view and action can only be chosen from:'''
        self.system_prompt += f'''{self.nodespace}\n'''
        self.system_prompt += f'''Do **not** include any other explanations or extra text in your output.
Remeber you must generate 5 actions and matched imagined_view in total! Think carefully of the objects you will see along the way!
'''

    def set_nodespace(self, nodespace: list):
        self.nodespace = nodespace
        self.set_system_prompt()

    def set_skillspace(self, skillspace: list):
        self.skillspace = skillspace
        #self.set_system_prompt()

    def load_detected_graph(self, graph: OptimizedTimeObjectGraph):
        self.detected_graph = graph

    def set_instruction(self, instruction: str):
        self.instruction = instruction

    def instruction_decomposer(self):
        # This method should decompose the instruction into sub goal graph.
        # 用于增强llm想象效果，暂时没使用
        pass

    def edit_user_prompt(self):
        self.user_prompt = f'''instruction: {self.instruction}\n'''
        self.user_prompt += "scene map:{"
        self.user_prompt += f'''{self.graph_to_string()} \n'''
        self.user_prompt += "}\n"
        self.user_prompt += f'''current position: {self.current_position}\n'''
    
    def graph_to_string(self):
        string_map = ""
        for time_value in self.detected_graph.time_nodes:
            time_id= self.detected_graph.time_nodes[time_value]
            objects_of_time = self.detected_graph.time_to_objects[time_id]
            #print(f"time_id = {time_id}, objects = {objects}")
            string_map += f"{list(objects_of_time)}\n"
        print(f"object-group {string_map}")
        return string_map

    def get_llm_response(self):
        message= json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": '{}'.format(self.system_prompt)
                },
                {
                    "role": "user", 
                    "content": '{}'.format(self.user_prompt)
                }
        ],
            "temperature": self.temperature
        })
        
        connection = http.client.HTTPSConnection(self.url)
        connection.request("POST", "/v1/chat/completions", message, self.headers)
        response = connection.getresponse()
        data = response.read()
        dict = json.loads(data.decode("utf-8"))
        return dict["choices"][0]["message"]["content"]

    def response_extractor(self, response):
        lines = response.strip().splitlines()
        #print(response)
        #print('*'*50)
        #print(lines)
        self.action_chain = []
        self.imagined_graph_chain = []
        try:
            # 逐行解析
            for line in lines:
                if not line or '{' not in line:
                    continue 
                # 替换单引号为双引号，使其符合 JSON 格式
                line_fixed = line.replace("'", '"')
                line_fixed = line_fixed.replace("\n", '').replace("”", '"')
                data = json.loads(line_fixed)

                if "action" in data:
                    self.action_chain.append((data["action"]["num_of_order"], data["action"]["object"]))
                elif "imagined_view" in data:
                    self.imagined_graph_chain.append(data["imagined_view"])

            # 根据 num_of_order 排序 action
            self.action_chain.sort(key=lambda x: x[0])
            target_chain = [obj for _, obj in self.action_chain]

            # 输出结果
            #print("target_chain =", target_chain)
            #print("self.imagined_graph_chain =")
            #print(self.imagined_graph_chain)
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            sys.exit(1) 

    def update_imagined_graph(self):
        self.predicted_graph = self.detected_graph
        self.edit_user_prompt()
        #print(self.system_prompt)
        #print(self.user_prompt)
        response=self.get_llm_response()
        self.response_extractor(response)
        num_steps = len(self.imagined_graph_chain)
        max_time_id =3#设为固定值仅用于测试，实际需要另行获取
        for i in range(num_steps):
            imagined_view = self.imagined_graph_chain[i]
            if imagined_view:
                self.predicted_graph.add_recognition(max_time_id+i,imagined_view)
        # self.predicted_graph.visualize('predicted_graph')

    def rethinking(self):
        """
        Rethink the action chain based on the current state and predicted graph.
        This method can be used to adjust the action chain if the initial prediction is not feasible.
        """
        self.update_imagined_graph()
        return self.action_chain, self.imagined_graph_chain

if __name__ == "__main__":
    # Example usage
    # 模拟探测结果
    realgraph = OptimizedTimeObjectGraph()
    realgraph.add_recognition(0, ['chair', 'sofa', 'door', 'window', 'table'])
    realgraph.add_recognition(1, ['sofa', 'refrigerator', 'door', 'table', 'vase'])
    realgraph.add_recognition(2, ['door', 'tv', 'sink'])

    # 构建预测器
    predictor = Predictor()
    predictor.load_detected_graph(realgraph)#更新已探索图
    predictor.current_position = 'window'#设置当前位置
    #传入导航指令，从数据集获取，是一个字符串
    predictor.set_instruction("You are now at window, go to the tv in the living room, and then go to the toilet in the bathroom. Finally go to the plant in kitchen.")
    #做出预测
    predictor.update_imagined_graph()