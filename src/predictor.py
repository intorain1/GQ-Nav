
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
        self.object_list_path = os.path.join('/home/mspx/icra/GQ-Nav/src', 'processing_src', 'object_simplified.txt')
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

        self.system_prompt = "You are a smart human and you can imagine the spatial relation of object, predict unseen objects, as well as their spatial relation and the order of their appearance, according to current observation. Then navigate to the goal perfectly."
        self.json_string =""
        self.user_prompt =None
        #优化方向：现在的前几个goal只由instruction决定，并不合理，应该在路径中途细化想象；
        #skill没有体现，是一个空壳参数
    
    # 没有用上
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
        self.edit_user_prompt()  # 编辑用户提示

    def instruction_decomposer(self):
        # This method should decompose the instruction into sub goal graph.
        # 用于增强llm想象效果，暂时没使用
        pass


    def edit_user_prompt(self,is_rethinking=False):
        # self.user_prompt = f'''instruction: {self.instruction}\n'''
        # if is_rethinking:
        #     current_local_detected_graph = self.detected_graph.time_to_objects[self.detected_graph.get_time_values()[-1]]
        #     # print(f"current_local_detected_graph = {current_local_detected_graph}")
        #     self.user_prompt += f'''History information: Based on the instruction, you have given an ordered series of predicted actions and their corresponding imagined_view as follows\n'''
        #     self.user_prompt += f'''{self.response}\n'''
        #     self.user_prompt += f'''Now you have successfully reached the first {self.error_action_num-1} target objects in the action chain.\n'''
        #     self.user_prompt += f'''But in actual situation, the agent can't {self.error_object} at "action_num {self.error_action_num}". The previos predicted view{self.error_imagined_graph} is far different from real observation {current_local_detected_graph}''' # 错误反馈
        #     self.user_prompt += f'''What should the agent do next? Now you need to rethink next actions and views. Give the action chain based on the current state, error information above and the following real scene map.\n'''

        # self.user_prompt += "scene map:{"
        # self.user_prompt += f'''{self.graph_to_string()} \n'''
        # self.user_prompt += "}\n"
        # self.user_prompt += f'''current position: {self.current_position}\n'''

        # if is_rethinking:
        #     self.user_prompt += f'''Rethink and predict the next actions and corresponding imagined_view. Number the actions start from 1.\n'''
        
        self.user_prompt = f'''Your task is to navigate with instruction {self.instruction}, and the objects in your environment are among {self.nodespace}.\n '''
        if is_rethinking is False:
            if len(self.detected_graph.time_nodes) > 0:
                self.user_prompt += f'''Now you have explored some parts of the scene and known the following objects in your environment: {self.graph_to_string()}. 
                The objects in a pair of [] are the objects detected in the same viewpoint. The time order is from top to bottom. The first line is the objects detected at the first viewpoint, and so on. The last line is currently in the agent's observation view.\n'''
            self.user_prompt += f'''What should the agent do next? Now you need to rethink next actions and views. You should use given information of ego state of agent, detected objects, current observation and position to analyze the waypoints, thinking in the following steps to finish this task.
            1. Fisrt, you should break down the instruction to at least 3 subtasks, for example: first go into xxx room, then xxx way, then … .\n'''
        if is_rethinking:
            self.user_prompt += f'''You have explored some parts of the scene and detected some "known objects" in your environment: {self.graph_to_string()}.
              The objects in a pair of [] are the objects detected in the same viewpoint. The time order is from top to bottom. The first line is the objects detected at the first viewpoint, and so on. The last line is currently in the agent's observation view.\n'''
            self.user_prompt =f'''Now you are halfway through the navigation task. Your previous prediction is {self.json_string}.'''
            self.user_prompt += f''' So far, you have successfully reached the first {self.error_action_num} goal objects.'''
            self.user_prompt += f''' But when trying to get to "{self.error_object}", the real observation of objects(i.e. the last line of "known objects") differs a lot from your previous prediction.\n'''
            self.user_prompt += f'''What should the agent do next? Now you need to rethink  next actions and views. Your answers shoulf not repeat the "group" and "goal" in already-done subtasks!! Just tell me your plan of next steps. You should use given information of prediction error, ego state of agent, detected objects, current observation and position to analyze the waypoints, thinking in the following steps to finish this task.
            1. Fisrt, you should break down the instruction to subtasks, and reason about the subtasks waiting to be done. For example: Subtasks in instruction is …. According to previous successful steps, xxx tasks were done. Then, according to the error information, first undo xxx step. Next, first go into xxx room, then xxx way, then … .\n'''
        self.user_prompt += f'''2. Then reasoning: in your experience, what objects would you find in the rooms/hallway/… in the first step? Please give me a set of objects that you would expect to find in each subtasks respectively. Each subtask should have a set of at least 4 objects.
        3. Next, What are the objects you would find when transferring from one room/hallay/… to another? Predicted the objects to help the agent to complete the instruction. Notice that there should be at least 2 transferring movements between subtasks. The objects in the transferring process should include at least 1 object respectively that is in the 2 neighbor room/hallway/… you mentioned in step 2.
        
        The second and third steps generate a series of object "group" in this task. The number of "group" is added from the number of rooms/ways and the number of transferring movements. How many "group" do you think there are in this task?
        For example, if there are 3 rooms and 2 transferring movements, then there will be 5 "group" in total.
        4. Then organize the relative all objects of each "group" above in chronological order and output the result as a 2-dim array in python.
        Objects can be repeatable. If one type of objects appear twice in the process of complete the task, its name should also appear twice in the array. Remember, all the metioned objects should be in the list:{self.nodespace}.
        5. Finally, choose at leasst 1 object in each "group" as the "goal". Total number of goals from all groups should be STRICTLY more than 7 !!! What combination of "goal" objects is the most possible to lead the agent to complete the instruction?
        '''

        if is_rethinking:
            self.user_prompt += f'''The goal objects for "undo step" can't be any goal objects that have be mentioned in "previous prediction".\n'''

        self.user_prompt+=f'''6. In the end, output the result in format of a list of dicts with two keys for each stage: "group" and "goal". The format must be like this:
        [
            {{
                "group": ["object1", "object2", ...],
                "goal": "object_goal"
            }},
            {{
                "group": ["object1", "object2", ...],
                "goal": "object_goal"
            }},
            ...
        ]
        This list must can be decoded as a json object.
        Tell me your results of all 6 steps one by one! Thank you a lot! 
        '''

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
        if self.user_prompt is not None:
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
        else:
            print("No user prompt!")
            sys.exit(1)

    def response_extractor(self):
        try:
            end_idx = self.response.rfind(']')
            stack = []
            json_string = ""
            for i in range(end_idx, -1, -1):
                if self.response[i] == ']':
                    stack.append(']')
                elif self.response[i] == '[':
                    stack.pop()
                    if not stack:
                        # 找到了与最后一个 ] 匹配的 [
                        json_string = self.response[i:end_idx+1]
                        break
            # json_string = json_string.partition('[')[2]
            # json_string = '[' + json_string 
            # json_string = json_string.rpartition(']')[0] + ']'
<<<<<<< HEAD
            self.json_string = json_string.replace("'", '"')  # 替换单引号
=======
            json_string = json_string.replace("'", '"')  # 替换单引号
>>>>>>> e942a97be9c0a64cea957f8393867642875cfaf2
            data = json.loads(json_string.strip())
            
            # 4. 使用列表推导式高效地提取group和goal
            self.imagined_graph_chain = [item['group'] for item in data]
            self.imagined_graph_chain[-1].append('STOP')  
            self.action_chain = [item['goal'] for item in data]
            num_steps = len(self.imagined_graph_chain)
            for i in range(num_steps):
                if i == num_steps - 1:
                    self.action_chain[i] = (i, self.action_chain[i], "STOP")
                else:
                    self.action_chain[i] = (i, self.action_chain[i], "NORMAL")  # 添加动作序号和状态

        except IndexError as e:
            print("错误：在文本中未找到 'answer6:' 或数组越界。")
            print("异常信息：", e)
            print("当前 response 内容：\n", self.response)
            return [], []
        except json.JSONDecodeError as e:
            print("错误：'answer6:'之后的内容不是有效的JSON格式。")
            print("异常信息：", e)
            print("当前待解析 json_string：\n", json_string)
            print("当前 response 内容：\n", self.response)
            return [], []
        except KeyError as e:
            print("错误：JSON对象中缺少键:", e)
            print("当前 data 内容：\n", data if 'data' in locals() else 'data 未定义')
            print("当前 response 内容：\n", self.response)
            return [], []
        except Exception as e:
            print("未知错误：", e)
            print("当前 response 内容：\n", self.response)
            return [], []
        # if self.response is None:
        #     print("No response to extract.")
        # else:
        #     lines = self.response.strip().splitlines()
        #     #print(response)
        #     #print('*'*50)
        #     #print(lines)
        #     self.action_chain = [(0,self.current_position,"START")]  # 初始化动作链，包含起始动作
        #     self.imagined_graph_chain = [['START']] #当前视角的想象内容初始化start
        #     try:
        #         # 逐行解析
        #         max_num_of_order = 0
        #         for line in lines:
        #             if not line or '{' not in line:
        #                 continue 
        #             # 替换单引号为双引号，使其符合 JSON 格式
        #             line_fixed = line.replace("'", '"').replace("\\","")
        #             line_fixed = line_fixed.replace("\n", '').replace("”", '"')
        #             if line_fixed.endswith('},'):
        #                 line_fixed = line_fixed[:-2] + '}'
        #             data = json.loads(line_fixed)

        #             if "action" in data:
        #                 self.action_chain.append((data["action"]["num_of_order"], data["action"]["object"],"NORMAL"))
        #                 if data["action"]["num_of_order"] > max_num_of_order:
        #                     max_num_of_order = data["action"]["num_of_order"]
        #             elif "imagined_view" in data:
        #                 self.imagined_graph_chain.append(data["imagined_view"])
        #                 self.imagined_graph_chain[-1]  # 添加停止动作
        #                 # imagined_view 的总长度比action_chain少1
        #         # print("decomposed imagined_graph_chain =", self.imagined_graph_chain)
        #         # 根据 num_of_order 排序 action
        #         self.action_chain.sort(key=lambda x: x[0])
        #         self.action_chain[-1]=(self.action_chain[-1][0],self.action_chain[-1][1],"STOP") # 添加停止动作
        #         self.imagined_graph_chain[-1].append('STOP')  # 最后一个动作的想象内容为空

        #         # 输出结果
        #         #print("target_chain =", target_chain)
        #         #print("self.imagined_graph_chain =")
        #         #print(self.imagined_graph_chain)
        #     except json.JSONDecodeError as e:
        #         print(f"JSON Decode Error: {e}")
        #         sys.exit(1) 

    def update_imagined_graph(self, is_rethinking=False):
        '''
        structure of action_chain:         [(0, 'current_position', 'START'),  (1, 'door','NORMAL'),         (2, 'tv','NORMAL'),             (3, 'sink','STOP')]
        structure of imagined_graph_chain: [['START'],                         ['door', 'window', 'table'],  ['tv', 'sofa', 'refrigerator'],  ['sink', 'vase', 'STOP']]
        '''
        self.predicted_graph = self.detected_graph._copy()
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
        # print("action_chain =", self.action_chain)
        # print("imagined_graph_chain =", self.imagined_graph_chain)
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
        self.error_imagined_graph = self.imagined_graph_chain[self.error_action_num]  # 获取错误动作对应的想象图
        self.update_imagined_graph(is_rethinking=True)  # Update the imagined graph based on the current state
        return self.action_chain, self.imagined_graph_chain

if __name__ == "__main__":
    # Example usage
    # 模拟探测结果
    realgraph = OptimizedTimeObjectGraph()
    realgraph.add_recognition(0, ['door', 'window', 'shelf', 'cabinet', 'vent'])
    realgraph.add_recognition(1, ['window', 'window', 'door'])
    realgraph.add_recognition(2, ['mattress', 'window', 'picture', 'shelf', 'cabinet', 'bed', 'stairs', 'vent'])

    # 构建预测器
    predictor = Predictor()
    predictor.load_detected_graph(realgraph)#更新已探索图
    #传入导航指令，从数据集获取，是一个字符串
    predictor.set_instruction("Go through the door way and down the hallway, turning left at the end looking into a bedroom with a gray throw on the bed.")
    #做出预测
    predictor.update_imagined_graph()
    print("response =\n", predictor.response)
    print("action_chain =", predictor.action_chain)
    print("imagined_graph_chain =", predictor.imagined_graph_chain)
    error_point= predictor.action_chain[-2]  # 假设倒数第二个动作是错误的
    print("error_point =", error_point)
    print("*"*50)
    renewed_graph = OptimizedTimeObjectGraph()
    renewed_graph.add_recognition(0, ['chair', 'sofa', 'door', 'window', 'table'])
    renewed_graph.add_recognition(1, ['sofa', 'refrigerator', 'door', 'table', 'vase'])
    renewed_graph.add_recognition(2, ['door', 'tv', 'sink'])
    renewed_graph.add_recognition(3, ['window', 'pillow', 'pillow', 'pillow', 'pillow', 'shelf', 'cabinet', 'chandelier', 'vent', 'pool', 'console',  'table', 'picture', 'frame', 'vent'])  # 假设在错误动作后看到了这些物体
    renewed_graph.add_recognition(4,['mattress', 'window', 'picture', 'shelf', 'cabinet', 'bed', 'stairs', 'vent'])
    predictor.load_detected_graph(renewed_graph)  # 更新探测图
    print("rethinking...")
    predictor.rethinking(error_action=error_point)
    print("action_chain after rethinking :", predictor.action_chain)
    print("imagined_graph_chain after rethinking :", predictor.imagined_graph_chain)


    ##TODO: goal是子集的命令调试验证
    ### RESULT：很不稳定，经常不在子集中；手动加入？
    ##TODO: 预测效果验证
    ### RESULT：有时预测的步骤多于所需，给出多余的步骤导航阶段数量————估计需要动态“stop”（每一步都做与终点想象图的相似度计算？开销大吗）
    ##TODO: skill暂时没有用到导航中
    ##TODO: self.current_position没有与探测/任务初始化/current_step保持同步更新，完整程序中必须修改后才可以开始预测或者rethink!
    ## optional:
    ## 逐步提问CoT————现在感觉是有必要的：根据指令要预测几步？->终点物体 ->终点处场景 ->过程中场景
