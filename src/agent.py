"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""
import json
import yaml
import re
import warnings
import time
import numpy as np
from collections import deque
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Dict, Union
from env import R2RNavBatch
from argparse import Namespace
from agent_base import BaseAgent
from graph import OptimizedTimeObjectGraph
from semantic import SemanticSegmenter
from predictor import Predictor

class NavAgent(BaseAgent):
    def __init__(self, env: R2RNavBatch, args: Namespace):
        super().__init__(env)
        self.config = args
        self.env = env
        self.graph = OptimizedTimeObjectGraph()
        # self.segmenter = SemanticSegmenter()
        self.predictor = Predictor()
        self.nav_step = 0
        self.action_chain = deque()
        self.imagined_graph_chain = deque()
        self.stop = False
        self.threshold = -np.inf

    def parse_objects(self, objects: List[Dict[str, Dict[str, float]]]) -> List[Tuple[str, float, float]]:
        parsed = []
        for obj in objects:
            for name, props in obj.items():
                words = list(dict.fromkeys(name.split()))
                parsed.extend(words)
        return parsed
    
    def parse_navigable(self, navigable: List[Dict[str, Any]]) -> List[str]:
        parsed = []
        for nav in navigable:
            parsed.append(nav)
        return parsed
    
    def init_trajecotry(self, obs: List[dict]):
        """Initialize the trajectory with the given observation."""
        self.traj = [{
            'instr_id': obs['instr_id'],
            'path': [[obs['viewpoint']]],
        }]

    def _make_action(self, reset=True) -> str:
        if reset:  # Reset env
            cur_obs = self.env.reset()[0]
            # print(cur_obs['instruction'])
        else:
            cur_obs = self.env._get_obs()[0]
        # print(cur_obs)

        self.init_trajecotry(cur_obs)
        objects = self.parse_objects(cur_obs['objects'])
        self.graph.add_recognition(self.nav_step, objects)
        self.predictor.load_detected_graph(self.graph)
        self.predictor.set_instruction(cur_obs['instruction'])

        while(not self.stop):
        # Update graph
            # self.graph.visualize('1')
            # print(len(self.action_chain))
            # # Judge if use action_chain
            if len(self.action_chain) == 0:
                self.predictor.update_imagined_graph()
                self.action_chain = deque(self.predictor.action_chain)
                self.imagined_graph_chain = deque(self.predictor.imagined_graph_chain)
                self.imagined_graph_chain_copy = self.imagined_graph_chain.copy()
                step = self.nav_step
                # print(self.imagined_graph_chain)
                self.imagined_graph = OptimizedTimeObjectGraph()
                while(len(self.imagined_graph_chain_copy) != 0):
                    image_graph = self.imagined_graph_chain_copy.popleft()
                    self.imagined_graph.add_recognition(step, image_graph)
                    step += 1
                # self.imagined_graph.visualize('2')
                # print('imagined_graph_chain', self.imagined_graph_chain)
                # print(self.imagined_graph_chain)
                # print('action', self.action_chain)

            #exe      
            action = self.action_chain.popleft()

            if action[2] == 'STOP':
                self.stop = True
                return self.traj

            to_object = action[1]
            # print(to_object)
            last_action = action
            
            # Get navigable candidates
            navigable = self.parse_navigable(cur_obs['candidate'])
            candidates = []
            candidate_graphs = []
            for candidate in navigable:
                nav_objects = self.parse_objects(self.env._get_object(cur_obs['scan'], candidate)['objects'])
                # print(nav_objects)
                candidate_graph = OptimizedTimeObjectGraph()
                candidate_graph.add_recognition(self.nav_step, nav_objects)
                candidate_graphs.append((candidate_graph, candidate))
                if to_object in nav_objects:
                    candidates.append((candidate_graph, candidate))
            
            ob = self.imagined_graph_chain.popleft()

            if len(candidates) == 0:  # not in all candidates
                max_score = -np.inf
                destination = None
                graph = OptimizedTimeObjectGraph()
                graph.add_recognition(self.nav_step, ob)
                for candidate_graph, candidate in candidate_graphs:
                    score = candidate_graph.match_score(graph, 0, 1, 0)
                    if score > max_score:
                        max_score = score
                        destination = candidate

            elif len(candidates) > 2:
                max_score = -np.inf
                destination = None
                graph = OptimizedTimeObjectGraph()
                graph.add_recognition(self.nav_step, ob)
                for candidate_graph, candidate in candidates:
                    score = candidate_graph.match_score(graph, 0, 1, 0)
                    if score > max_score:
                        max_score = score
                        destination = candidate

            else:
                destination = candidates[0][1]

            # print(f"agent-destination {destination} ")
            self.env.step([destination])# the parameter is a list of next viewpoint IDs. Change destination to a list
            self.traj[0]['path'].append([destination])

            self.nav_step += 1
            cur_obs = self.env._get_obs()[0]
            objects = self.parse_objects(cur_obs['objects'])
            self.graph.add_recognition(self.nav_step, objects)
            # print(objects)

            # now_graph = OptimizedTimeObjectGraph()
            # now_graph.add_recognition(self.nav_step, objects)

            # imagine_graph = OptimizedTimeObjectGraph()
            # imagine_graph.add_recognition(self.nav_step, ob)
            # self.graph.visualize('3')
            # self.imagined_graph.visualize('4')
            score = self.graph.match_score(self.imagined_graph, 0, 1, 0)
            print(score)
            if  score <= self.threshold:
                self.predictor.load_detected_graph(self.graph)
                action_chain_list, imagined_graph_chain_list = self.predictor.rethinking(last_action)
                self.action_chain = deque(action_chain_list)
                self.imagined_graph_chain = deque(imagined_graph_chain_list)
                self.imagined_graph_chain_copy = self.imagined_graph_chain.copy()
                step = self.nav_step
                self.imagined_graph = OptimizedTimeObjectGraph()
                while(len(self.imagined_graph_chain_copy) != 0):
                    image_graph = self.imagined_graph_chain_copy.popleft()
                    self.imagined_graph.add_recognition(step, image_graph)
                    step += 1
                
            else:
                self.threshold = score

        
        return self.traj
                # print(self.traj[0]['path'])
        # operation of deque progress
        # if len(self.action_chain) > 0:
        #     self.action_chain.popleft()
        #     self.imagined_graph_chain.popleft()

    def _visulize(self):
        pass

    

                


                

        