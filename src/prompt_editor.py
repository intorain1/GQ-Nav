import re
import math
from typing import List
from graph import OptimizedTimeObjectGraph
from openai import OpenAI
import sys
import os
import http.client
import json

# 每次调用前应该维护trajectory current_candidate_objects
class PromptEditor:
    def __init__(self,args,instruction:str,start_object:str,optgraph:OptimizedTimeObjectGraph):
        self.args=args

        self.start_object=start_object
        self.instruction=instruction
        self.history=[]
        self.trajectory=[]
        self.trajectory.append(self.start_object)
        self.planning=["Navigation has just started, with no planning yet."]

        self.object_list_path = os.path.join('src', 'processing_src', 'object_simplified.txt')
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

        # graph info
        self.opt_graph=optgraph._copy()
        self.norm_graph=self.set_graph(self.opt_graph)
        self.current_candidate_objects=[]
    
    def get_llm_response(self,system_prompt='',user_prompt='Hello world!'):
        if self.user_prompt is not None:
            message= json.dumps({
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": '{}'.format(system_prompt)
                    },
                    {
                        "role": "user", 
                        "content": '{}'.format(user_prompt)
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
    
    def set_candidate_objects(self,current_observed_objects:List[str]):
        self.current_candidate_objects=current_observed_objects

    def set_trajectory(self,reached_object:str):
        self.trajectory.append(reached_object)

    def set_graph(self, optgraph:OptimizedTimeObjectGraph):
        """
        Set the graph for the prompt manager.
        :param graph: OptimizedTimeObjectGraph instance, size: batch_size
        """
        self.opt_graph = optgraph._copy()
        self.graph= self.optgraph_to_normgraph()

    def optgraph_to_normgraph(self):
        """
        Suit OptimizedTimeObjectGraph to the format of
        {
            'vp1': ['vp2', 'vp3'],   # vp1 可到达 vp2 和 vp3
            'vp2': ['vp4', 'vp5'],   # vp2 可到达 vp4 和 vp5
            ...
        }
        """
        graph_dict = {}

        current_nodes_list_in_obj = []
        for obj in self.opt_graph.get_objects():
            graph_dict[obj] = list(self.opt_graph.object_to_times[obj])
            neighbors = set()
            if graph_dict[obj] is None:
                continue
            for time_id in graph_dict[obj]:
                for neighbor in self.opt_graph.time_to_objects[time_id]:
                    if neighbor != obj:
                        neighbors.add(neighbor)
            graph_dict[obj] = list(neighbors)

        return graph_dict

    def make_r2r_prompts(self, cand_inputs, time_id):

        background = """You are an embodied robot that navigates in the real world."""
        background_supp = """You need to explore between some places marked with certain objects and ultimately find the destination to stop.""" \
        + """ At each step, a series of objects and their relations corresponding to the places you have explored and have observed will be provided to you."""\
        

        instr_des = """'Instruction' is a global, step-by-step detailed guidance, but you might have already executed some of the commands. You need to carefully discern the commands that have not been executed yet."""
        # newly added
        normal_obj="""'Detected-object' is any object that you have seen in your exploration. """
        obj_destination="""'Object-destination' is an object chosen from Detected-objects. An object-destination is the object you have moved to or need to appoarch in next step."""

        traj_info = """'Trajectory' represents the object-destination info of the objects you have moved to. You start navigating from {self.start_object}."""
        map_info = """'Map' refers to the connectivity between the detected-objects."""
        map_supp = """'Supplementary Info' records some detected-objects you have ever seen but have not yet been chosen as object-destination. These markers are only considered when there is a navigation error, and you decide to backtrack for further exploration."""
        history = """'History' represents the places you have explored in previous steps along with their corresponding images. It may include the correct landmarks mentioned in the 'Instruction' as well as some past erroneous explorations."""
        option = """'Action options' are some actions that you can take at this step."""
        pre_planning = """'Previous Planning' records previous long-term multi-step planning info that you can refer to now."""

        requirement = """For each provided image of the places, you should combine the 'Instruction' and carefully examine the relevant information, such as scene descriptions, landmarks, and objects. You need to align 'Instruction' with 'History' (including corresponding images) to estimate your instruction execution progress and refer to 'Map' for path planning. Check the Place IDs in the 'History' and 'Trajectory', avoiding repeated exploration that leads to getting stuck in a loop, unless it is necessary to backtrack to a specific place."""
        dist_require = """If you can already see the destination, estimate the distance between you and it. If the distance is far, continue moving and try to stop within 1 meter of the destination."""
        thought = """Your answer must include four parts: 'Thought', 'Distance', 'New Planning', and 'Action'. You need to combine 'Instruction', 'Trajectory', 'Map', 'Supplementary Info', your past 'History', 'Previous Planning', 'Action options', and your life experience of indoor scene to think about what to do next and why, and complete your thinking into 'Thought'."""
        new_planning = """Based on your 'Map', 'Previous Planning' and current 'Thought', you also need to update your new multi-step path planning to 'New Planning'."""
        action = """At the end of your output, you must provide a single capital letter in the 'Action options' that corresponds to the action you have decided to take, and place only the letter into 'Action', such as "Action: A"."""

        task_description = f"""{background} {background_supp}\n{instr_des}\n{normal_obj}\n{obj_destination}\n{history}\n{traj_info}\n{map_info}\n{map_supp}\n{pre_planning}\n{option}\n{requirement}\n{dist_require}\n{thought}\n{new_planning}\n{action}"""

        init_history = 'The navigation has just begun, with no history.'
        action_options, only_options = self.make_action_options(cand_inputs, time_id=time_id)

        instruction =self.instruction

        trajectory_text, graph_text, graph_supp_text = self.make_map_prompt()

        if time_id == 0:
            prompt = f"""Instruction: {instruction}\nHistory: {init_history}\nTrajectory: {trajectory_text}\nMap:{graph_text}\nSupplementary Info: {graph_supp_text}\nPrevious Planning:\n{self.planning[-1]}\nAction options (step {str(time_id)}): {action_options}"""
        else:
            prompt = f"""Instruction: {instruction}\nHistory: {self.history}\nTrajectory: {trajectory_text}\nMap:{graph_text}\nSupplementary Info: {graph_supp_text}\nPrevious Planning:\n{self.planning[-1]}\nAction options (step {str(time_id)}): {action_options}"""


        nav_input = {
            "task_description": task_description,
            "prompts" : prompt,
            "only_options": only_options,
            "action_options": action_options,
            "only_actions": cand_inputs["action_prompts"]
        }

        return nav_input

    def make_action_options(self, cand_inputs, time_id):

        action_prompts = cand_inputs["action_prompts"]

        action_prompts = action_prompts
        if bool(self.args.stop_after):
            if time_id >= self.args.stop_after:
                action_prompts = ['stop'] + action_prompts

        action_options = [chr(j + 65)+'. '+action_prompts[j] for j in range(len(action_prompts))]
        only_options = [chr(j + 65) for j in range(len(action_prompts))]

        return action_options, only_options

# 原nodes_list在现在和trajectory统一了
    def make_action_prompt(self,loaded_candidate_objects=None):
        cand_objects = []
        action_prompts = []
        cur_candidates=[]
        # cand views
        if loaded_candidate_objects is None:
            cur_candidates=self.current_candidate_objects
        else:
            cur_candidates=loaded_candidate_objects

        for j, obj in enumerate(cur_candidates):

            cand_objects.append(obj)
            action_text = f" Go to the place of object-destination: {obj}"
            action_prompts.append(action_text)

        return {
            'cand_objects': cand_objects,
            'action_prompts': action_prompts,
        }

    # 每执行一步需要调用一次，在agent中调用
    # a_t传入刚执行完的动作索引号！'A/B/C/D'-65
    def make_history(self, a_t, nav_input, time_id):
        nav_input["only_actions"] = ['stop'] + nav_input["only_actions"]
        last_action = nav_input["only_actions"][a_t]
        if time_id == 0:
            self.history += f"""step {str(time_id)}: {last_action}"""
        else:
            self.history += f""", step {str(time_id)}: {last_action}"""

    def make_map_prompt(self):
        # graph-related text
        trajectory = self.trajectory
        graph = self.graph

        no_dup_nodes = []
        trajectory_text = 'The place of object-destination'
        graph_text = ''

        candidate_nodes = graph[trajectory[-1]]

        # trajectory and map connectivity
        for node in trajectory:
            trajectory_text += f""" {node}"""
            if node not in no_dup_nodes:
                no_dup_nodes.append(node)

                adj_text = ''
                adjacent_nodes = graph[node]
                for adj_node in adjacent_nodes:
                    adj_text += f""" {adj_node},"""

                graph_text += f"""\nObject-destination {node} is connected with detected-objects {adj_text}"""[:-1]

        # ghost nodes info
        graph_supp_text = ''
        supp_exist = None
        for _, node in enumerate(trajectory):

            if node in trajectory or node in candidate_nodes:
                continue
            supp_exist = True
            graph_supp_text += f"""\nPlace of destination-object: {node}"""

        if supp_exist is None:
            graph_supp_text = """Nothing yet."""

        return trajectory_text, graph_text, graph_supp_text

if __name__=="__main__":
    candidates=[]
    time_id=0
    start_object='window'
    instruction=""
    class Args:
        batch_size = 1
        stop_after = 5
        max_action_len = 10
        response_format = 'str'
        llm = 'gpt-4-vision-preview'
        max_tokens = 100
        
    realgraph = OptimizedTimeObjectGraph()
    realgraph.add_recognition(0, ['door', 'window', 'shelf', 'cabinet', 'vent'])
    realgraph.add_recognition(1, ['window', 'window', 'door'])
    realgraph.add_recognition(2, ['mattress', 'window', 'picture', 'shelf', 'cabinet', 'bed', 'stairs', 'vent'])
    candidates=['mattress', 'window', 'picture', 'shelf', 'cabinet', 'bed', 'stairs', 'vent']

    args = Args()    
    promptManager=PromptEditor(args=args,start_object=start_object,instruction=instruction,optgraph=realgraph._copy())
    prompt=promptManager.make_r2r_prompts(cand_inputs=candidates,time_id=time_id)
    print(prompt)
    response=promptManager.get_llm_response(user_prompt=prompt)

#TODO: self.planning的更新还没做