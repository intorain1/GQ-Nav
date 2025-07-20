import os
import json
import time

from env_src.utils.data_utils import construct_instrs
from env_src.utils.logger import write_to_record_file

from env_src.utils.data import ImageObservationsDB
from parser import parse_args
from env import R2RNavBatch
from agent import NavAgent
import argparse

def build_dataset(args):

    feat_db = ImageObservationsDB(args.obj_dir)

    dataset_class = R2RNavBatch

    val_env_names = [args.val_env_name]

    val_envs = {}
    for split in val_env_names:
        val_instr_data = construct_instrs(
            args.anno_dir, args.dataset, [split]
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, args.navigable_dir,
            batch_size=args.batch_size, seed=args.seed, name=split,
        )   # evaluation using all objects
        val_envs[split] = val_env

    return val_envs

if __name__ == "__main__":

    args = parse_args()
    # print(args.anno_dir, args.dataset, args.val_env_name)
    val_envs = build_dataset(args)
    agent = NavAgent(next(iter(val_envs.values())), args)
    for i in range(0,5):
        if i == 0:
            agent._make_action(threshold=0.5)  # Example threshold
        else:
            agent._make_action(threshold=0.5, reset=False)
    # agent.rollout(reset=True)
    # agent.rollout(reset=False)
    
