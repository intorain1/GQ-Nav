from graph import OptimizedTimeObjectGraph
from openai import OpenAI
import os
import http.client
import json
import sys
import random

system_prompt = 'you are a smart human and you can navigate to the goal perfectly'
user_prompt = 'now you should navigate with instruction{instr}, and the object in your environment is {objects}, so you should analyze the waypoint to finish' \
            'this task, like fisrt you should go into xxx room, then xxx room/way. in your experience, what object would you find in the xxx room/way? please give me'

class Predictor:
    def __init__(self):
        # skill和object闭集配置
        self.skillspace = ["move", "go_upstairs", "go_downstairs", "open_door"]
        self.object_list_path = '/home/mspx/icra/GQ-Nav/src/processing_src/object.txt'
        with open(self.object_list_path, 'r') as f:
            self.nodespace = [line.strip() for line in f if line.strip()]
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
        self.response= None  # LLM response
        self.action_chain=[] #用物体进行描述
        self.imagined_graph_chain = []

        # 错误断点
        self.error_action_num = None
        self.error_object = None  # 错误动作的目标对象
        self.error_skill = None
        self.error_imagined_graph = None

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
Given the current scene map, the target instruction, history information and your current position, your job is to:
- Generate the next several actions in order , leading the robot to gradually approach the target.
- Each action must specify the navigation skill used and the the target object reached.

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
    "num_of_order": <1-the number of stages>,
    "skill": <one of [move, go_upstairs, go_downstairs, open_door]>,
    "object": <name of the target object reached>
}
}

After each action, **imagine a object-group** you can see after reaching the taget position according to the instruction and the current scene map.
Prediction of each action must be expressed as an object-group:
{"imagined_view": ['bed', 'oven', 'door', 'sink']}
Notice that the object-group must include the **target object of the action** and several **other objects** that can be seen from the target position.

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
Most importantly! The objects in the imagined_view and action_chain can only be chosen from:'''
        self.system_prompt += f'''{self.nodespace}. And the target object must be subset of the imagined_view.'''
        self.system_prompt += f'''Do **not** include any other explanations or extra text in your output.
Remeber you must generate enough steps of actions and matched imagined_view to completely express the requirements in the instruction and make sure the agent can gradually approach the goal!! Think carefully of the objects you will see along the way!
'''

        self.user_prompt =""
        #优化方向：现在的前几个goal只由instruction决定，并不合理，应该在路径中途细化想象；
        #skill没有体现，是一个空壳参数
    
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
        pass

    def set_nodespace(self, nodespace: list):
        # self.nodespace = nodespace
        # self.set_system_prompt()
        pass

    def set_skillspace(self, skillspace: list):
        # self.skillspace = skillspace
        # #self.set_system_prompt()
        pass

    def load_detected_graph(self, graph: OptimizedTimeObjectGraph):
        self.detected_graph = graph

    def set_current_position(self, position: str):
        self.current_position = position

    def set_instruction(self, instruction: str):
        self.instruction = instruction

    def instruction_decomposer(self):
        # This method should decompose the instruction into sub goal graph.
        # 用于增强llm想象效果，暂时没使用
        pass


    def edit_user_prompt(self,is_rethinking=False):
        self.user_prompt = f'''instruction: {self.instruction}\n'''
        if is_rethinking:
            current_local_detected_graph = self.detected_graph.time_to_objects[self.detected_graph.get_time_values()[-1]]
            # print(f"current_local_detected_graph = {current_local_detected_graph}")
            self.user_prompt += f'''History information: Based on the instruction, you have given an ordered series of predicted actions and their corresponding imagined_view as follows\n'''
            self.user_prompt += f'''{self.response}\n'''
            self.user_prompt += f'''Now you have successfully reached the first {self.error_action_num-1} target objects in the action chain.\n'''
            self.user_prompt += f'''But in actual situation, the agent can't {self.error_object} at "action_num {self.error_action_num}". The previos predicted view{self.error_imagined_graph} is far different from real observation {current_local_detected_graph}''' # 错误反馈
            self.user_prompt += f'''What should the agent do next? Now you need to rethink next actions and views. Give the action chain based on the current state, error information above and the following real scene map.\n'''

        self.user_prompt += "scene map:{"
        self.user_prompt += f'''{self.graph_to_string()} \n'''
        self.user_prompt += "}\n"
        self.user_prompt += f'''current position: {self.current_position}\n'''

        if is_rethinking:
            self.user_prompt += f'''Rethink and predict the next actions and corresponding imagined_view. Number the actions start from 1.\n'''
    
    def graph_to_string(self):
        string_map = ""
        for time_value in self.detected_graph.time_nodes:
            time_id= self.detected_graph.time_nodes[time_value]
            objects_of_time = self.detected_graph.time_to_objects[time_id]
            #print(f"time_id = {time_id}, objects = {objects}")
            string_map += f"{list(objects_of_time)}\n"
        # print(f"object-group {string_map}")
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
        self.response = dict["choices"][0]["message"]["content"]
        return dict["choices"][0]["message"]["content"]

    def response_extractor(self):
        if self.response is None:
            print("No response to extract.")
        else:
            lines = self.response.strip().splitlines()
            #print(response)
            #print('*'*50)
            #print(lines)
            self.action_chain = [(0,self.current_position,"START")]  # 初始化动作链，包含起始动作
            self.imagined_graph_chain = [['START']] #当前视角的想象内容初始化start
            try:
                # 逐行解析
                max_num_of_order = 0
                for line in lines:
                    if not line or '{' not in line:
                        continue 
                    # 替换单引号为双引号，使其符合 JSON 格式
                    line_fixed = line.replace("'", '"').replace("\\","")
                    line_fixed = line_fixed.replace("\n", '').replace("”", '"')
                    if line_fixed.endswith('},'):
                        line_fixed = line_fixed[:-2] + '}'
                    data = json.loads(line_fixed)

                    if "action" in data:
                        self.action_chain.append((data["action"]["num_of_order"], data["action"]["object"],"NORMAL"))
                        if data["action"]["num_of_order"] > max_num_of_order:
                            max_num_of_order = data["action"]["num_of_order"]
                    elif "imagined_view" in data:
                        self.imagined_graph_chain.append(data["imagined_view"])
                        self.imagined_graph_chain[-1]  # 添加停止动作
                        # imagined_view 的总长度比action_chain少1
                # print("decomposed imagined_graph_chain =", self.imagined_graph_chain)
                # 根据 num_of_order 排序 action
                self.action_chain.sort(key=lambda x: x[0])
                self.action_chain[-1]=(self.action_chain[-1][0],self.action_chain[-1][1],"STOP") # 添加停止动作
                self.imagined_graph_chain[-1].append('STOP')  # 最后一个动作的想象内容为空

                # 输出结果
                #print("target_chain =", target_chain)
                #print("self.imagined_graph_chain =")
                #print(self.imagined_graph_chain)
            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                sys.exit(1) 

    def update_imagined_graph(self, is_rethinking=False):
        '''
        structure of action_chain:         [(0, 'current_position', 'START'),  (1, 'door','NORMAL'),         (2, 'tv','NORMAL'),             (3, 'sink','STOP')]
        structure of imagined_graph_chain: [['START'],                         ['door', 'window', 'table'],  ['tv', 'sofa', 'refrigerator'],  ['sink', 'vase', 'STOP']]
        '''
        self.predicted_graph = self.detected_graph
        self.edit_user_prompt(is_rethinking=is_rethinking)  # 编辑用户提示
        #print(self.system_prompt)
        #print(self.user_prompt)
        self.response=self.get_llm_response()
        # print("origin response=",self.response)
        self.response_extractor()
        # print("extracted action_chain =", self.action_chain)
        # print("extracted imagined_graph_chain =", self.imagined_graph_chain)
        num_steps = len(self.imagined_graph_chain)
        max_time_id = max(self.detected_graph.get_time_values())
        for i in range(num_steps):
            imagined_view = self.imagined_graph_chain[i]
            if imagined_view:
                self.predicted_graph.add_recognition(max_time_id+i,imagined_view)
        # self.predicted_graph.visualize('predicted_graph')

    def rethinking(self, error_action):
        # rethink 前需要load_detected_graph
        # must pop from completed action_chain and imagined_graph_chain
        """
        Rethink the action chain based on the current state and predicted graph.
        This method can be used to adjust the action chain if the initial prediction is not feasible.
        """
        '''
        new_structure (to be defined):
        In previous resoning, the action chain is:{previous_action_chain}
        Now you have successfully reached the first {num_steps} target objects in the action chain.
        '''
        self.error_action_num = error_action[0] # 获取错误动作的序号
        self.error_object = error_action[1]  # 获取错误动作的目标对象
        self.error_imagined_graph = self.imagined_graph_chain[self.error_action_num][:-1]  # 获取错误动作对应的想象图
        self.update_imagined_graph(is_rethinking=True)  # Update the imagined graph based on the current state
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
    print("action_chain =", predictor.action_chain)
    error_point= predictor.action_chain[-2]  # 假设倒数第二个动作是错误的
    print("error_point =", error_point)
    print("*"*50)
    predictor.set_current_position(error_point[1])  # 更新当前位置为错误动作的目标对象
    renewed_graph = OptimizedTimeObjectGraph()
    renewed_graph.add_recognition(0, ['chair', 'sofa', 'door', 'window', 'table'])
    renewed_graph.add_recognition(1, ['sofa', 'refrigerator', 'door', 'table', 'vase'])
    renewed_graph.add_recognition(2, ['door', 'tv', 'sink'])
    renewed_graph.add_recognition(3, ['sink', 'oven', 'microwave', 'refrigerator'])  # 假设在错误动作后看到了这些物体
    predictor.load_detected_graph(renewed_graph)  # 更新探测图
    print("rethinking...")
    predictor.rethinking(error_action=error_point)

    ##TODO: goal是子集的命令调试验证
    ### RESULT：很不稳定，经常不在子集中；手动加入？
    ##TODO: 预测效果验证
    ### RESULT：有时预测的步骤多于所需，给出多余的步骤导航阶段数量————估计需要动态“stop”（每一步都做与终点想象图的相似度计算？开销大吗）
    ##TODO: skill暂时没有用到导航中
    ##TODO: self.current_position没有与探测/任务初始化/current_step保持同步更新，完整程序中必须修改后才可以开始预测或者rethink!
    ## optional:
    ## 逐步提问CoT————现在感觉是有必要的：根据指令要预测几步？->终点物体 ->终点处场景 ->过程中场景
