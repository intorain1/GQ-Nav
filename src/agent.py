"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""
import json
import yaml
import re
import warnings
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
        self.path = []

    def parse_objects(self, objects: List[Dict[str, Dict[str, float]]]) -> List[Tuple[str, float, float]]:
        parsed = []
        for obj in objects:
            for name, props in obj.items():
                parsed.append(name)
        return parsed
    
    def parse_navigable(self, navigable: List[Dict[str, Any]]) -> List[str]:
        parsed = []
        for nav in navigable:
            parsed.append(nav)
        return parsed

    def _make_action(self, threshold, reset=True) -> str:
        if reset:  # Reset env
            cur_obs = self.env.reset()[0]
        else:
            cur_obs = self.env._get_obs()[0]

        # print(cur_obs)
        # Update graph
        objects = self.parse_objects(cur_obs['objects'])
        self.graph.add_recognition(self.nav_step, objects)
        self.predictor.load_detected_graph(self.graph)
        self.nav_step += 1

        # print(len(self.action_chain))
        # # Judge if use action_chain
        if len(self.action_chain) == 0:
            self.predictor.set_instruction(cur_obs['instruction'])
            self.predictor.update_imagined_graph()
            self.action_chain = deque(self.predictor.action_chain)
            self.imagined_graph_chain = deque(self.predictor.imagined_graph_chain)
            # print(self.imagined_graph_chain)
            # print('action', self.action_chain)

        # elif self.graph.match_score(self.imagined_graph_chain.popleft(), 0, 0, 0) <= threshold:
        #     action_chain_list, imagined_graph_chain_list = self.predictor.rethinking()
        #     self.action_chain = deque(action_chain_list)
        #     self.imagined_graph_chain = deque(imagined_graph_chain_list)

        # Get target object
        to_object = self.action_chain.popleft()[1]
        # print(to_object)

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

        if len(candidates) == 0:  # not in all candidates
            max_score = -np.inf
            destination = None
            graph = OptimizedTimeObjectGraph()
            ob = self.imagined_graph_chain.popleft()
            graph.add_recognition(self.nav_step, ob)
            for candidate_graph, candidate in candidate_graphs:
                score = candidate_graph.match_score(graph, 0, 1, 0)
                if score > max_score:
                    max_score = score
                    destination = candidate

        elif len(candidates) > 2:
            max_score = -np.inf
            destination = None
            for candidate_graph, candidate in candidates:
                score = candidate_graph.match_score(self.imagined_graph_chain[0], 0, 1, 0)
                if score > max_score:
                    max_score = score
                    destination = candidate

        else:
            destination = candidates[0][1]

        self.path.append(destination)
        print(f"agent-destination {destination} ")
        self.env.step([destination])# the parameter is a list of next viewpoint IDs. Change destination to a list

        # operation of deque progress
        # if len(self.action_chain) > 0:
        #     self.action_chain.popleft()
        #     self.imagined_graph_chain.popleft()

    def _visulize(self):
        pass

    def _ifstop(self):
        # isreach?
        pass

    def _rollout(self, **args):
        pass
                


                

        