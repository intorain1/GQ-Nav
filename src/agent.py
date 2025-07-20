"""Agent that interacts with Matterport3D simulator via a hierarchical planning approach."""
import json
import yaml
import re
import warnings
import numpy as np
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
        self.action_chain = []
        self.imagine_chain = []
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
        self.nav_step += 1

        # # Judge if use action_chain
        if len(self.action_chain) == 0:
            self.predictor.set_instruction(cur_obs['instruction'])
            self.predictor.update_imagined_graph()
            self.action_chain = self.predictor.action_chain
            print(self.action_chain)

        elif self.graph.match_score(self.imagine_chain[0], 0, 0, 0) <= threshold:
            self.action_chain, self.imagine_chain = self.predictor.rethinking()

        # Get target object
        to_object = self.action_chain[0]

        # Get navigable candidates
        navigable = self.parse_navigable(cur_obs['candidate'])
        candidates = []
        candidate_graphs = []
        for candidate in navigable:
            nav_objects = self.parse_objects(self.env._get_object(cur_obs['scan'], candidate)['objects'])
            candidate_graph = OptimizedTimeObjectGraph()
            candidate_graph.add_recognition(self.nav_step, nav_objects)
            candidate_graphs.append((candidate_graph, candidate))
            if to_object in nav_objects:
                candidates.append((candidate_graph, candidate))

        if len(candidates) == 0:  # not in all candidates
            max_score = -np.inf
            destination = None
            for candidate_graph, candidate in candidate_graphs:
                score = candidate_graph.match_score(self.imagine_chain[0], 0, 0, 0)
                if score > max_score:
                    max_score = score
                    destination = candidate

        elif len(candidates) > 2:
            max_score = -np.inf
            destination = None
            for candidate_graph, candidate in candidates:
                score = candidate_graph.match_score(self.imagine_chain[0], 0, 0, 0)
                if score > max_score:
                    max_score = score
                    destination = candidate

        else:
            destination = candidates[0][1]

        self.path.append(destination)
        self.env.step(destination)

    def _visulize(self):
        pass

    def _ifstop(self):
        pass

    def _rollout(self, **args):
        pass
                


                

        