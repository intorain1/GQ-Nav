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
        self.segmenter = SemanticSegmenter()
        self.predictor = Predictor()
        self.nav_step = 0
        self.action_chain = []
        self.imagine_chain = []
        self.path = []

    def _make_action(self, threshold) -> str:
        cur_obs = self.env._get_obs()[0]

        ##update graph
        objects = cur_obs['objects']
        self.graph.add_recognition[self.nav_step, objects]
        self.nav_step += 1

        ##judge if use action_chain
        if self.graph.match_score(self.imagine_chain[0], 0, 0, 0) <= threshold:
            self.action_chain, self.image_chain = self.predictor.rethinking()
    
        ##get objects
        to_object = self.action_chain[0]

        ##get navigable
        navigable = cur_obs['navigable']
        candidates = []
        candidate_graphs = []
        for candidate in navigable:
            nav_objects = self.env.get_object(cur_obs['scan'], candidate)
            candidate_graph = OptimizedTimeObjectGraph()
            candidate_graph.add_recognition(self.nav_step, nav_objects)
            candidate_graphs.append(candidate_graph, candidate)
            if to_object in nav_objects:
                candidates.append(candidate_graph, candidate)
        
        if len(candidates) == 0: ## not in all candidates
            max = -np.inf
            for candidate_graph, candidate in candidate_graphs:
                score = candidate_graph.match_score(self.imagine_chain[0], 0, 0, 0)
                if score > max:
                    max = score
                    destination = candidate

        elif len(candidates) > 2:
            max = -np.inf
            for candidate_graph, candidate in candidates:
                if candidate_graph.match_score(self.imagine_chain[0], 0, 0, 0) > max:
                    max = candidate_graph.match_score(self.imagine_chain[0], 0, 0, 0)
                    destination = candidate
        
        else:
            destination = candidates[0][1]

        self.path.append(destination)
        self.env.step(destination)

    def _visulize(self):
        pass

                


                

        